r"""Where in the past the source informed the future, read without the biases that make it wrong.

The lag-resolved KL attribution is the readout this architecture exists to produce:

$$\widetilde K_{t,\ell} = \sum_m K^{(m)}_t\,\alpha^{(m)}_{t,\ell},
\qquad \sum_\ell \widetilde K_{t,\ell} = K_t.$$

That identity is exact rather than approximate -- each head's attention sums to one over its
valid lags, and the latent groups are head-aligned, so the split is a decomposition rather than a
weighting. It is also fragile in one specific way, and this analysis re-measures it on the run
being reported rather than inheriting it from a fixture: dropout on the attention probabilities
would rescale them so they no longer sum to one, and the attribution would then hold only in
expectation. Every number below would still look entirely reasonable.

**The profile is reported three times and the three may disagree.** The raw attribution divides
every lag bin by the same anchor total, which is what keeps it summing to $\bar K$ and makes it a
decomposition; it is also biased short whenever lag $\ell$ is causally valid only at anchors
$t \ge \ell$, because the long lags are then averaged over anchors that could not have contributed
to them. The support-corrected profile divides each bin by its own contributing-anchor count. The
untruncated one restricts to the anchors at which every lag exists. Where two argmaxes disagree,
the difference *is* the corresponding bias.

**At this cell's shipped geometry all three coincide, and that is measured rather than assumed.**
The anchor floor is $F = 133$ while the furthest searched lag is $L - 1 = 90$, so every lag is
causally valid at every scored anchor and all three corrections collapse to no-ops. But the three
quantities behind that -- the floor, ``max_lag`` and ``lag_floor`` -- move independently, and a
``sweep_floor_*`` arm would reintroduce truncation with nothing saying so. So this analysis reads
preflight's own ``lag_support_margin_steps``, measures the per-lag contributing-anchor counts, and
records whether the two agree. It never asserts the simplification: a truncated-support run is a
legitimate geometry, and what would not be legitimate is reporting the untruncated reading while
the geometry had stopped supporting it.

**An argmax alone is not a reading of a profile.** Ties resolve to the lowest index, and
``entmax15`` -- which the shipped model uses -- assigns lags exactly zero, so a profile that is
flat or nearly empty still has a perfectly confident argmax. So the peak is described rather than
merely located: its width, the mass concentrated near it, whether a second peak exists, and
whether the profile is degenerate by a stated mechanical criterion rather than by eye. That
vocabulary now lives in :mod:`~teb_vae.lag_attn_cfs.eval.lag_shape` rather than here, because
``lag_clocks`` reports a per-segment peak against the same criterion and an analysis may not import
another; it is re-exported under its original names, so nothing about reading this module changed.

**The lag is the compensated one, and the axis is stored-coefficient time.** $\tau = 4(\ell +
\delta)$ seconds, with $\delta$ read from the model's own accessor and converted by the module both
the training figure and this analysis share. What the axis is time *in* is the caveat that matters:
the coefficients come from a strictly one-sided bank whose composed per-channel group delay reaches
the same order as the lag search itself, so a peak's position here is not a physiological latency
and is not a transfer entropy.
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT` therefore travels on the artifacts
that state a lag *position* -- the two peak tables, the figure and the summary block -- rather than
in a document beside them. It is deliberately **not** repeated down the two per-lag profile tables:
those are the input to a positional reading rather than a reading, they run to thousands of rows on
the stratified axis, and a four-hundred-character sentence per row would be tens of megabytes of
one sentence.

**Under a channel alignment this analysis still needs $\delta$ and not the reference.** Every
seconds column here is built through
:func:`~teb_vae.lag_attn_cfs.eval.lag_axis.compensated_seconds_axis` or the scalar converter beside
it, both of which produce the stored-coefficient axis; the alignment's own constant
$\tau_{\mathrm{ref}}$ is a physical delay that would turn that axis into a physiological one, which
is a claim this analysis does not make. It is recorded as ``source_reference_delay_s`` in the run's
causality disclosure and its summary, where a reader who wants it can find it without any table
here quietly having applied it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import cohort
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels
from teb_vae.lag_attn_cfs.eval.frames import (
    grouped_frame_entry,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_cfs.eval.lag_axis import (
    GROUP_DELAY_CAVEAT,
    compensated_seconds_axis,
    padded_profile,
    profile_column,
    read_lag_support,
)
# The peak vocabulary, bound rather than owned. It was defined here until ``lag_clocks`` needed to
# report a per-segment peak and could not reach an analysis to get the guard that makes one
# readable; it now lives one layer down and is re-exported under these names, so this module's own
# call sites and every reader of ``lag_kl.degeneracy`` are unchanged. Two copies of a threshold
# would be two definitions of what a positional claim is allowed to mean.
from teb_vae.lag_attn_cfs.eval.lag_shape import (  # noqa: F401  (re-exported)
    DEGENERATE_PEAK_TO_MEDIAN,
    DEGENERATE_ZERO_FRACTION,
    PEAK_FRACTION,
    SECONDS_PER_LAG_STEP,
    degeneracy,
    mass_above,
    peak_width,
    secondary_peaks,
)
from teb_vae.lag_attn_cfs.eval.report_seam import IDENTITY_TOLERANCE, identity_tolerance_for
from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "lag_kl"

#: What it writes.
PROFILE_FILENAME = "lag_kl_profile.csv"
SUMMARY_FILENAME = "lag_kl_summary.csv"
PER_RECORDING_FILENAME = "lag_kl_per_recording.csv"
STRATIFIED_PROFILE_FILENAME = "lag_kl_stratified_profile.csv"
STRATIFIED_PEAKS_FILENAME = "lag_kl_stratified_peaks.csv"

#: The figure, named as ``FIGURE_GUIDE.md`` names it.
PROFILE_FIGURE = "lag_kl_profile"

#: The per-sample columns this analysis reduces per recording -- the identity residual, so a
#: reader can see which recordings carry it, and the KL the profile decomposes.
VALUE_COLUMNS: Tuple[str, ...] = (
    "lag_map_identity_max_abs",
    "head_kl_identity_max_abs",
    "source_conditioned_kl_raw",
)


#: The three profiles, as ``(reported name, the lag block's key, the reported column, what it is)``.
#: Three rather than the sibling's two: the untruncated recomputation is retained here and reported
#: rather than folded into the corrected one, because whether the three coincide is the measurement
#: that says this geometry admits every lag at every anchor.
PROFILES: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "raw",
        "kl_lag_profile",
        "kl_nats",
        "divides every bin by the same anchor total; sums over lags to the headline KL and "
        "is biased toward short lags wherever a lag is not valid at every scored anchor",
    ),
    (
        "support_corrected",
        "kl_lag_profile_support_corrected",
        "kl_nats_support_corrected",
        "divides each bin by its own contributing-anchor count; does not sum to the KL and "
        "is the profile to read for where the source informed",
    ),
    (
        "untruncated",
        "kl_lag_profile_untruncated",
        "kl_nats_untruncated",
        "restricted to the anchors at which every lag exists; free of the renormalisation the "
        "correction above cannot reach, and identical to the other two when the measured lag "
        "support margin is non-negative",
    ),
)


def profile_frame(lag: Dict[str, Any], seconds: np.ndarray) -> pd.DataFrame:
    """Lay the three profiles and their denominator out as one table, keyed by lag.

    Args:
        lag: The pass's lag block.
        seconds: The compensated-seconds axis.

    Returns:
        One row per lag: its index, its compensated seconds, the raw attribution and its share of
        the total, the support-corrected and untruncated attributions, and the
        contributing-anchor count that separates the first two.
    """
    columns = ["lag_step", "compensated_seconds", "kl_nats", "share"]
    columns += [column for _, _, column, _ in PROFILES if column != "kl_nats"]
    columns += ["anchor_count"]

    raw = np.asarray(list(lag.get("kl_lag_profile") or []), dtype=np.float64)
    if raw.size == 0:
        return pd.DataFrame(columns=columns)
    counts = np.asarray(list(lag.get("kl_lag_anchor_counts") or []), dtype=np.float64)
    total = float(raw.sum())
    frame = pd.DataFrame(
        {
            "lag_step": np.arange(raw.size, dtype=int),
            "compensated_seconds": seconds[: raw.size],
            "kl_nats": raw,
            "share": raw / total if total > 0.0 else np.full(raw.size, np.nan),
        }
    )
    for _, key, column, _ in PROFILES:
        if column == "kl_nats":
            continue
        frame[column] = padded_profile(
            np.asarray(list(lag.get(key) or []), dtype=np.float64), raw.size
        )
    frame["anchor_count"] = padded_profile(counts, raw.size)
    return frame.reindex(columns=columns)


def build_summary_rows(lag: Dict[str, Any], delay_steps: int) -> List[Dict[str, Any]]:
    """Describe each profile's peak, one row per profile.

    Args:
        lag: The pass's lag block.
        delay_steps: The causal input delay, for the seconds column.

    Returns:
        One row per profile -- raw, support-corrected and untruncated -- carrying the argmax and
        its seconds, the peak's width and edges, the mass concentrated near it, how many secondary
        peaks were found, and the degeneracy verdict. A row whose profile is degenerate still
        reports its argmax; what changes is that the row says not to read it.
    """
    rows: List[Dict[str, Any]] = []
    for name, key, _column, note in PROFILES:
        profile = list(lag.get(key) or [])
        peak = peak_width(profile)
        concentration = mass_above(profile)
        secondary = secondary_peaks(profile)
        verdict = degeneracy(profile)
        argmax = peak["argmax"]
        rows.append(
            {
                "profile": name,
                "source_key": key,
                "meaning": note,
                "argmax_lag_step": argmax,
                "compensated_seconds": (
                    None if argmax is None
                    else float(lag_compensated_seconds(argmax, delay_steps=delay_steps))
                ),
                "peak_nats": peak["peak"],
                "peak_lo_lag_step": peak["lo"],
                "peak_hi_lag_step": peak["hi"],
                "peak_width_bins": peak["width_bins"],
                "peak_width_seconds": (
                    None if peak["width_bins"] is None
                    else float(peak["width_bins"]) * float(SECONDS_PER_LAG_STEP)
                ),
                "mass_above_half_peak": concentration["share"],
                "n_bins_above_half_peak": concentration["n_bins"],
                "n_secondary_peaks": len(secondary),
                "secondary_peak_lag_steps": [record["lag_step"] for record in secondary],
                "degenerate": verdict["degenerate"],
                "peak_to_median": verdict["peak_to_median"],
                "zero_fraction": verdict["zero_fraction"],
                "degenerate_reasons": "; ".join(verdict["reasons"]),
                # On the row that states a lag position, so a reader who opens this table alone
                # cannot take ``compensated_seconds`` for a physiological latency.
                "axis_caveat": GROUP_DELAY_CAVEAT,
            }
        )
    return rows


# =================================================================================================
# The measured lag support
# =================================================================================================
#: How far apart two profiles may sit before they are reported as disagreeing, in nats per anchor.
#: The three are the *same* reduction under three denominators when every lag is valid everywhere,
#: so their difference is float summation order and nothing else -- but they are accumulated over
#: different orderings, so an exact test would fail on arithmetic rather than on geometry.
PROFILE_AGREEMENT_TOLERANCE = 1e-9


def measured_lag_support(lag: Dict[str, Any], recorded: Dict[str, Any]) -> Dict[str, Any]:
    r"""Compare what preflight computed about the lag support against what the pass observed.

    Two independent statements, and reporting only one of them is how a simplification outlives
    the geometry that justified it:

    * **Computed.** ``lag_support_margin_steps`` $= \min_t \mathcal A - (L-1) - F_u$, from the
      checkpoint's own geometry. Non-negative means every lag is causally valid at every scored
      anchor.
    * **Observed.** The per-lag contributing-anchor counts this pass accumulated. Uniform across
      lags means the same thing, measured on the data rather than derived from three config keys.

    Nothing here raises and nothing is asserted. A truncated-support run is a legitimate geometry
    that the support-corrected and untruncated profiles exist to handle; what this records is
    whether the two statements agree, so a disagreement is a number in the output rather than a
    reading nobody checked.

    Args:
        lag: The pass's lag block, for the anchor counts and the three profiles.
        recorded: What :func:`~teb_vae.lag_attn_cfs.eval.lag_axis.read_lag_support` returned.

    Returns:
        The recorded block, the observed uniformity and the counts behind it, the largest
        disagreement between the three profiles, and whether the computed and observed readings
        agree. ``None`` wherever a quantity was not measured, never a default.
    """
    counts = np.asarray(list(lag.get("kl_lag_anchor_counts") or []), dtype=np.float64)
    finite = counts[np.isfinite(counts)]
    uniform = (
        None if finite.size == 0
        else bool(np.max(finite) - np.min(finite) <= PROFILE_AGREEMENT_TOLERANCE)
    )

    profiles = [
        padded_profile(lag.get(key) or [], int(counts.size or 0))
        for _, key, _, _ in PROFILES
    ]
    spread = float("nan")
    if counts.size and len(profiles) > 1:
        stacked = np.vstack(profiles)
        if np.isfinite(stacked).all():
            spread = float(np.max(np.abs(stacked - stacked[0])))
    agree = None if not np.isfinite(spread) else bool(spread <= PROFILE_AGREEMENT_TOLERANCE)

    expected = recorded.get("every_lag_valid_at_every_anchor")
    return {
        **recorded,
        "anchor_counts_uniform": uniform,
        "min_anchor_count": None if finite.size == 0 else float(finite.min()),
        "max_anchor_count": None if finite.size == 0 else float(finite.max()),
        "profiles_agree": agree,
        "max_abs_profile_difference": spread,
        "agreement_tolerance": PROFILE_AGREEMENT_TOLERANCE,
        # The two readings of one property. A mismatch means the geometry preflight described and
        # the geometry the pass decoded at are not the same geometry, which no other number here
        # would show.
        "computed_and_observed_agree": (
            None if expected is None or uniform is None else bool(expected == uniform)
        ),
    }


# =============================================================================
# Stratified profiles
#
# No pipeline before this one cut the lag readout by anything. What is emitted here is the whole
# 91-bin profile per cohort and per time window rather than a per-cohort argmax, because an argmax
# is not a reading of a profile -- and because two cohorts whose peaks coincide can still put very
# different amounts of mass near them.
#
# The profiles stratified are the ones free of a bias the pooled reading has to state and remove:
# the support-corrected pair, whose per-lag denominator is each lag's own contributing-anchor
# count, and the **untruncated** pair, restricted to the anchors at which every lag exists. Only
# the second is free of the renormalisation the correction cannot reach, so a per-cohort argmax
# claim rests on it -- which is why the restricted anchor range travels in the output beside every
# row.
# =============================================================================
#: The per-sample vector readouts stratified, as ``(reported profile, attribute, what it is)``.
#: Every one travels the same aggregation chain the pooled profiles do -- per recording first,
#: then across recordings within the cohort -- so a per-cohort profile and the pooled one differ
#: only in which recordings entered them.
STRATIFIED_PROFILES: Tuple[Tuple[str, str, str], ...] = (
    (
        "kl_support_corrected",
        "lag_profile_support_corrected",
        "per-lag KL attribution on each lag's own contributing-anchor count",
    ),
    (
        "kl_untruncated",
        "lag_profile_untruncated",
        "per-lag KL attribution over the anchors whose lag support is complete -- the profile a "
        "per-cohort argmax claim rests on",
    ),
    (
        "attention_support_corrected",
        "attention_profile_support_corrected",
        "head-averaged attention on each lag's own contributing-anchor count",
    ),
    (
        "attention_untruncated",
        "attention_profile_untruncated",
        "head-averaged attention over the anchors whose lag support is complete",
    ),
)

#: The per-head vector, kept separate because it is $M \cdot L$ wide and is reshaped rather than
#: read directly. Head-averaging before profiling discards exactly what the head-structured
#: posterior exists to expose, so the per-cohort reading keeps the heads apart too.
PER_HEAD_ATTRIBUTE = "attention_profile_per_head"

#: The axis name the time-window stratification is reported under. Not a cohort column on the
#: table -- it is derived from ``epoch`` -- so it is named here rather than looked up.
TIME_AXIS = "time_window"


def _is_null_label(value: Any) -> bool:
    """Return whether a cohort label is absent, in either of the two forms it arrives in.

    A label collected in-process is ``None``; the same label read back out of ``per_sample.csv``
    is ``NaN``. Both mean the segment belongs to no cohort.

    Args:
        value: One row's label on one axis.

    Returns:
        ``True`` when the label is absent.
    """
    return value is None or (isinstance(value, float) and np.isnan(value))


def _group_profiles(
    rows: np.ndarray, guids: Sequence[Any], groups: Sequence[Any], axis: str
) -> Dict[str, Tuple[np.ndarray, int]]:
    """Average a per-sample vector within each recording, then within each cohort.

    Args:
        rows: The per-sample vectors, $(n, C)$, in the per-sample table's row order.
        guids: The recording of each row.
        groups: The cohort of each row; rows with no cohort are dropped, because a segment with no
            cohort belongs to none of them and folding them together would create one named after
            the absence.
        axis: The stratification axis, choosing the returned order. Both call sites lay the result
            straight into the emitted rows, so this is what puts the stratified table in the same
            worst-first cohort order every figure in this evaluation is drawn in; ``groupby``
            alone would order it alphabetically. On the time axis, which the canonical order does
            not know, the alphabetical order stands and ``bin_center_h`` carries the number a
            reader sorts on.

    Returns:
        Cohort to ``(profile, n_recordings)``. ``NaN`` rows -- the segments that scored no anchors
        -- are skipped by the means rather than imputed, so a cohort's count is the recordings
        that actually measured something.
    """
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(guids) or values.shape[0] != len(groups):
        return {}
    frame = pd.DataFrame(values)
    frame["guid"] = list(guids)
    # Stringifying before the null test would turn a NaN label into the cohort `"nan"`, which
    # `notna()` then keeps: the table is read back from CSV on any re-run, and that is where an
    # absent class or a non-canonical shard basename arrives as NaN rather than as `None`.
    frame["group"] = [None if _is_null_label(value) else str(value) for value in groups]
    frame = frame[frame["group"].notna()]
    if frame.empty:
        return {}
    columns = list(range(values.shape[1]))
    per_guid = frame.groupby(["group", "guid"])[columns].mean()
    per_group = per_guid.groupby("group")[columns].mean()
    # Recordings that measured something, not rows: a recording whose every segment scored no
    # anchors is an all-NaN row that the means above correctly skip, so counting it here would
    # label the profile with evidence that did not go into it.
    counts = per_guid.notna().any(axis=1).groupby("group").sum().astype(int)
    profiles = {
        str(group): (
            np.asarray(per_group.loc[group], dtype=np.float64), int(counts.loc[group])
        )
        for group in per_group.index
    }
    return {name: profiles[name] for name in cohort.ordered_groups(list(profiles), axis)}


def _axis_labels(per_sample: pd.DataFrame) -> Dict[str, Tuple[List[Any], Dict[str, float]]]:
    """Return each stratification axis's per-row labels, plus the time axis's window centres.

    Args:
        per_sample: The collected per-sample table.

    Returns:
        Axis name to ``(labels, centres)``. ``centres`` is empty except on the time axis, where it
        maps each window's label to its centre in hours so the emitted rows carry a number as well
        as a name.
    """
    axes: Dict[str, Tuple[List[Any], Dict[str, float]]] = {
        axis: ([value for value in per_sample[axis]] if axis in per_sample.columns else [], {})
        for axis in labels.GROUP_COLUMNS
    }
    if "epoch" not in per_sample.columns:
        return axes

    # The same 0.5 h grid the trajectory analysis is cut on, taken from the layer below rather
    # than restated: two analyses reading different grids would report windows that do not line up.
    binned = cohort.add_time_bins(per_sample)
    centres: Dict[str, float] = {}
    window: List[Any] = [None] * len(per_sample)
    # The binned frame is a row subset, so its index labels are the table's own -- which is what
    # puts each window label back on the row it came from rather than on the row at that position.
    positions = {label: position for position, label in enumerate(per_sample.index)}
    for label, centre in zip(binned.index, binned[cohort.BIN_CENTER_COLUMN]):
        name = f"{float(centre):g} h"
        centres[name] = float(centre)
        position = positions.get(label)
        if position is not None:
            window[position] = name
    axes[TIME_AXIS] = (window, centres)
    return axes


def stratified_profiles(
    per_sample: pd.DataFrame,
    vectors: Dict[str, np.ndarray],
    *,
    delay_steps: int,
    n_lags: int,
    num_heads: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Emit every stratified profile at full lag resolution, and record the axes that were skipped.

    Args:
        per_sample: The collected per-sample table, for the cohort labels and the time axis.
        vectors: The per-sample vector readouts, in the table's row order.
        delay_steps: The causal input delay, for the seconds column.
        n_lags: Lag window width $L$, for the per-head reshape.
        num_heads: How many heads the per-head vector holds.

    Returns:
        ``(frame, skipped)``. The frame is long-form -- one row per (axis, cohort, profile, lag) --
        so a reader pivots it rather than opening one file per cohort. ``skipped`` names each axis
        that carried fewer than two cohorts, which on a single-class split is the ordinary outcome
        and not a failure.
    """
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    guids = list(per_sample["guid"]) if "guid" in per_sample.columns else []
    requested: List[Tuple[str, str, str]] = list(STRATIFIED_PROFILES)
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, Any] = {}

    for axis, (row_labels, centres) in _axis_labels(per_sample).items():
        # The shared predicate, so an absent label cannot be counted as a cohort here and then
        # dropped by `_group_profiles` below -- which is how a single-class split kept its skip
        # note from being written.
        distinct = labels.distinct_groups(row_labels)
        if len(distinct) < 2:
            skipped[axis] = (
                f"{len(distinct)} distinct {axis} value(s) in this split, so there is nothing to "
                f"compare; the pooled profile stands"
            )
            continue
        for name, attribute, note in requested:
            for group, (profile, count) in _group_profiles(
                vectors.get(attribute, np.zeros((0, 0))), guids, row_labels, axis
            ).items():
                rows.extend(
                    _profile_rows(
                        axis, group, name, note, profile, seconds,
                        count=count, centre=centres.get(group),
                    )
                )
        rows.extend(
            _per_head_rows(
                axis, centres, vectors.get(PER_HEAD_ATTRIBUTE, np.zeros((0, 0))), guids,
                row_labels, seconds, n_lags=n_lags, num_heads=num_heads,
            )
        )

    columns = [
        "group_column", "group", "bin_center_h", "profile", "meaning", "head", "n_recordings",
        "lag_step", "compensated_seconds", "value",
    ]
    return pd.DataFrame(rows, columns=columns), skipped


def _profile_rows(
    axis: str,
    group: str,
    name: str,
    meaning: str,
    profile: np.ndarray,
    seconds: np.ndarray,
    *,
    count: int,
    centre: Optional[float] = None,
    head: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Lay one cohort's profile out as one row per lag."""
    values = padded_profile(profile, seconds.size)
    return [
        {
            "group_column": axis,
            "group": group,
            "bin_center_h": centre,
            "profile": name,
            "meaning": meaning,
            "head": head,
            "n_recordings": int(count),
            "lag_step": int(lag),
            "compensated_seconds": float(seconds[lag]),
            "value": float(values[lag]),
        }
        for lag in range(seconds.size)
    ]


def _per_head_rows(
    axis: str,
    centres: Dict[str, float],
    rows: np.ndarray,
    guids: Sequence[Any],
    row_labels: Sequence[Any],
    seconds: np.ndarray,
    *,
    n_lags: int,
    num_heads: int,
) -> List[Dict[str, Any]]:
    """Stratify the per-head attention, reshaping the flattened vector exactly once.

    The vector is flattened head-major so that it travels the same one-trailing-axis aggregation
    chain every other vector readout does. A flat vector whose length does not factor into
    $M \\cdot L$ is a mis-assembled profile rather than a short one, so it is dropped whole rather
    than reshaped into a plausible wrong answer.
    """
    values = np.asarray(rows, dtype=np.float64)
    if num_heads <= 0 or values.ndim != 2 or values.shape[1] != num_heads * n_lags:
        return []
    emitted: List[Dict[str, Any]] = []
    for group, (profile, count) in _group_profiles(values, guids, row_labels, axis).items():
        reshaped = profile.reshape(num_heads, n_lags)
        for head in range(num_heads):
            emitted.extend(
                _profile_rows(
                    axis, group, f"attention_head_{head}",
                    "per-head attention over the lags, on the pooled anchor support",
                    reshaped[head], seconds, count=count,
                    centre=centres.get(group), head=head,
                )
            )
    return emitted


def stratified_peak_rows(frame: pd.DataFrame, delay_steps: int) -> List[Dict[str, Any]]:
    """Describe each stratified profile's peak, one row per (axis, cohort, profile).

    Args:
        frame: The long-form stratified table.
        delay_steps: The causal input delay, for the seconds column.

    Returns:
        One row per profile carrying the same description the pooled summary carries -- the
        argmax and its compensated seconds, the peak's width and edges, the mass near it, the
        secondary peaks and the degeneracy verdict -- so a per-cohort reading is held to the same
        standard as the pooled one rather than being reported as a bare argmax. Ordered by axis,
        then by cohort in the canonical order, then by profile.
    """
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return rows
    # The canonical position of each cohort, per axis. Applied as a final sort rather than by
    # iterating in that order, so the row-building loop below stays a plain ``groupby``.
    position = {
        str(axis): {
            group: index
            for index, group in enumerate(
                cohort.ordered_groups(sorted({str(name) for name in cell["group"]}), str(axis))
            )
        }
        for axis, cell in frame.groupby("group_column", sort=True)
    }
    for (axis, group, name), cell in frame.groupby(
        ["group_column", "group", "profile"], sort=True
    ):
        ordered = cell.sort_values("lag_step")
        profile = list(np.asarray(ordered["value"], dtype=np.float64))
        peak = peak_width(profile)
        concentration = mass_above(profile)
        secondary = secondary_peaks(profile)
        verdict = degeneracy(profile)
        argmax = peak["argmax"]
        rows.append(
            {
                "group_column": axis,
                "group": group,
                "profile": name,
                "n_recordings": int(ordered["n_recordings"].iloc[0]),
                "argmax_lag_step": argmax,
                "compensated_seconds": (
                    None if argmax is None
                    else float(lag_compensated_seconds(argmax, delay_steps=delay_steps))
                ),
                "peak_value": peak["peak"],
                "peak_width_bins": peak["width_bins"],
                "peak_width_seconds": (
                    None if peak["width_bins"] is None
                    else float(peak["width_bins"]) * float(SECONDS_PER_LAG_STEP)
                ),
                "mass_above_half_peak": concentration["share"],
                "n_secondary_peaks": len(secondary),
                "secondary_peak_lag_steps": [record["lag_step"] for record in secondary],
                "degenerate": verdict["degenerate"],
                "degenerate_reasons": "; ".join(verdict["reasons"]),
                "axis_caveat": GROUP_DELAY_CAVEAT,
            }
        )
    rows.sort(
        key=lambda row: (
            row["group_column"],
            position.get(row["group_column"], {}).get(row["group"], len(rows)),
            row["profile"],
        )
    )
    return rows


def build_profile_figure(
    profile: pd.DataFrame, lag: Dict[str, Any], *, delay_steps: int, n_lags: int
) -> Any:
    """Draw the three profiles against the compensated-seconds axis, with the peak marked.

    Two panels. The top overlays all three attributions on the same axis, because the whole content
    of each correction is where it parts company with the raw one -- and at this cell's geometry
    the three lie on top of one another, which is itself the reading. The bottom is the
    contributing-anchor count that separates the first two -- drawn rather than described, since a
    reader looking at a corrected profile is entitled to see the denominator that produced it.

    The group-delay caveat is printed under the figure. A figure is the artifact most likely to be
    lifted out of a run directory and shown alone, and a peak read off one without it beside it is
    read as a physiological latency -- which is the one claim this axis cannot support.

    Args:
        profile: The per-lag table.
        lag: The pass's lag block, read for the two argmaxes.
        delay_steps: The causal input delay, for the axis.
        n_lags: Lag window width, so the axis spans the window even when the profile is empty.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    axis = axes[0, 0]
    figures.multi_line_panel(
        axis, seconds,
        np.vstack([profile_column(profile, column, n_lags) for _, _, column, _ in PROFILES]),
        [
            "raw attribution (sums to the KL)",
            "support-corrected (per contributing anchor)",
            "untruncated (anchors where every lag exists)",
        ],
        title="Per-lag KL attribution",
        xlabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        ylabel="nats per anchor",
    )
    for key, colour, label in (
        ("kl_argmax_lag_step", figures.COLOR_BLUE, "raw argmax"),
        ("kl_argmax_lag_step_support_corrected", figures.COLOR_ORANGE, "corrected argmax"),
    ):
        argmax = lag.get(key)
        if argmax is None:
            continue
        axis.axvline(
            float(lag_compensated_seconds(int(argmax), delay_steps=delay_steps)),
            color=colour, linestyle="--", linewidth=figures.LINE_REGULAR, label=label,
        )
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=figures.FONT_SMALL, loc="best", ncol=2)

    figures.multi_line_panel(
        axes[1, 0], seconds, profile_column(profile, "anchor_count", n_lags)[None, :],
        ["contributing anchors"],
        title="Anchors contributing to each lag -- the correction's denominator",
        xlabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        ylabel="anchors per segment",
    )
    figures.caveat_note(figure)
    return figure


def run_lag_kl_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the per-lag KL attribution, its peak structure, and the identity behind it.

    Args:
        context: The analysis context, read for the pass's lag block and the per-sample table.
        eval_config: The validated block. Unused: nothing here is tunable, because an operator who
            could widen the degeneracy criterion could make a flat profile read as a finding.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the peak description, the measured lag support, the identity
        residuals with the verdict they earn, and the delay every reported lag was compensated by.
    """
    collection = context.collection
    per_sample = collection.per_sample
    # The run's horizon, applied before anything is binned: it bounds the
    # population on the segment's own start, so every clock in the run answers
    # for the same segments. ``None`` leaves the frame untouched.
    per_sample = cohort.within_horizon(
        per_sample, eval_config.get("max_hours_before_delivery")
    )
    results = dict(getattr(collection, "results", None) or {})
    lag = dict(results.get("lag") or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    # Read off the run's own preflight record rather than derived from three config keys, so a
    # geometry arm that reintroduced truncation is a number here rather than a silent assumption.
    support = measured_lag_support(lag, read_lag_support(output_dir))

    delay_steps = int(lag.get("delay_steps") or 0)
    n_lags = int(lag.get("n_lags") or len(lag.get("kl_lag_profile") or []))
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    profile = profile_frame(lag, seconds)
    profile.to_csv(directory / PROFILE_FILENAME, index=False)

    summary_rows = build_summary_rows(lag, delay_steps)
    pd.DataFrame(summary_rows).to_csv(directory / SUMMARY_FILENAME, index=False)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    stratified, skipped_axes = stratified_profiles(
        per_sample,
        dict(getattr(collection, "vectors", None) or {}),
        delay_steps=delay_steps,
        n_lags=n_lags,
        num_heads=int(lag.get("num_heads") or 0),
    )
    stratified.to_csv(directory / STRATIFIED_PROFILE_FILENAME, index=False)
    stratified_peaks = stratified_peak_rows(stratified, delay_steps)
    pd.DataFrame(stratified_peaks).to_csv(directory / STRATIFIED_PEAKS_FILENAME, index=False)

    figure_name = str(
        figures.render_figure(
            build_profile_figure(profile, lag, delay_steps=delay_steps, n_lags=n_lags),
            directory / PROFILE_FIGURE,
        ).name
    )
    return {
        "n_samples": scored_sample_count(per_sample, "source_conditioned_kl_raw"),
        "composition": {"n_recordings": int(len(per_guid)), "n_lags": n_lags},
        "plan": {"capped": False},
        # Every reported lag carries both, because a lag index is not seconds and the delay it is
        # compensated by is a per-channel maximum rather than a single figure.
        "delay_steps": delay_steps,
        "source_delay_is_max_over_channels": bool(
            lag.get("source_delay_is_max_over_channels", True)
        ),
        # The sentence every lag-resolved artifact carries. In the record as well as under the
        # figure, because ``summary.json`` is the artifact that gets quoted and a reader of it
        # would otherwise have the lag numbers and no statement of what they are lags *in*.
        "axis_caveat": GROUP_DELAY_CAVEAT,
        # Measured, not assumed. What preflight computed from the geometry, what this pass observed
        # in its per-lag anchor counts, and whether the three profiles actually coincide.
        "lag_support": support,
        "peaks": summary_rows,
        # The stratified reading: which axes were cut, which were skipped for holding one cohort,
        # and the anchor restriction the untruncated profiles were computed over. The profiles
        # themselves are on the CSV at full lag resolution -- an argmax in a summary is exactly
        # the reduction this analysis exists to refuse.
        "stratified": {
            "axes": sorted(set(stratified["group_column"])) if len(stratified) else [],
            "skipped_axes": skipped_axes,
            "profiles": [name for name, _, _ in STRATIFIED_PROFILES],
            "n_rows": int(len(stratified)),
            "n_lags": n_lags,
            "restricted_to_anchors_from": max(n_lags - 1, 0),
            "peaks": stratified_peaks,
        },
        "identity": _identity_block(
            lag, (results.get("readouts") or {}).get("source_conditioned_kl_raw")
        ),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, VALUE_COLUMNS)
        ],
        "files": [
            PROFILE_FILENAME, SUMMARY_FILENAME, PER_RECORDING_FILENAME,
            STRATIFIED_PROFILE_FILENAME, STRATIFIED_PEAKS_FILENAME, figure_name,
        ],
    }


def _identity_block(lag: Dict[str, Any], kl_scale: Optional[float]) -> Dict[str, Any]:
    """Report both structural identities against the tolerance the run judges them at.

    Surfaced here as well as in the sanity block because this is the analysis whose every number
    rests on them: a lag profile that does not sum to the KL is a decomposition of nothing, and a
    reader holding this analysis's output should not have to go looking for whether it held. The
    tolerance is resolved through the same function the sanity block uses, so the two cannot
    reach opposite conclusions about one residual.

    Args:
        lag: The pass's lag block.
        kl_scale: The headline KL the identities are over, which sets the tolerance.

    Returns:
        The residuals, the tolerance and its floor, and whether each held. A residual the pass did
        not measure is absent rather than zero.
    """
    tolerance = identity_tolerance_for(kl_scale)
    block: Dict[str, Any] = {
        "tolerance_nats": float(tolerance),
        "tolerance_floor_nats": float(IDENTITY_TOLERANCE),
    }
    for key, value in dict(lag.get("identity_residuals") or {}).items():
        block[key] = float(value)
        block[f"{key}_holds"] = bool(float(value) <= tolerance)
    # The aggregate form of the second identity, which a reader can check by hand against the
    # headline KL: the per-head split is a decomposition of the same number.
    block["kld_per_head_total_nats"] = lag.get("kld_per_head_total_nats")
    return block
