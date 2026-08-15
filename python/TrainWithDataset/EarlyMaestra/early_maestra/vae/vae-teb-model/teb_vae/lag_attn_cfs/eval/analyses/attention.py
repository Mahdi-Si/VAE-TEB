r"""The attention itself: per head, against the entropy it can actually reach, unbiased by truncation.

Three things this reports that a head-averaged profile against $\log L$ cannot.

**Per head.** The posterior is head-structured: latent group $m$ is written by attention head $m$
alone, which is what makes the per-head KL an additive decomposition rather than an arbitrary
slice. Averaging the four heads before profiling discards exactly that -- four heads attending at
four different delays and one attending everywhere produce the same head-averaged profile. So the
per-head profiles, the per-head entropies and the per-head KL travel separately, and the
head-averaged profile is emitted beside them **named as such** rather than as "the" profile.

**Against the attainable ceiling, which is measured rather than assumed.** A distribution over $n$
outcomes has entropy at most $\log n$, and at anchor $t$ only $\min(t + 1, L)$ lags exist at all,
so the attainable ceiling is $\operatorname{mean}_t \log \min(t + 1, L)$ over the anchors actually
scored. On the raw cells that is strictly below $\log L$: their trained anchors start at $30$
against $L = 91$, so a quarter of them are structurally truncated and a model attending uniformly
over everything available to it reads as increasingly *concentrated* the earlier the anchor.

**Here it is not, and the difference is the anchor floor.** This cell decodes from $F = 133$ while
the furthest searched lag is $L - 1 = 90$, so every scored anchor sees the whole lag window and the
attainable ceiling is exactly $\log L$. That is a property of $F \ge L - 1$ rather than of the
domain: the floor, ``max_lag`` and ``lag_floor`` move independently, and a ``sweep_floor_*`` arm
would reintroduce truncation. So both ceilings are still computed and still reported distinctly,
their equality is **measured** against preflight's own ``lag_support_margin_steps``, and the
truncated-anchor count is derived from the model's geometry rather than quoted.

**Unbiased by truncation.** The support correction divides each lag bin by its own
contributing-anchor count, which fixes the denominator. It cannot fix the numerator: attention
rows are normalised per anchor, so at a truncated anchor the mass that had no long lag to reach
was renormalised onto the short ones. Nothing in a per-lag count knows that happened. The
restricted profile -- over the anchors at which every lag exists -- is what removes it, and the
restricted argmax is the only one an argmax *claim* should rest on. At a non-negative margin all
three coincide and the restriction costs nothing; at a negative one it costs the truncated anchors.
Either way all three profiles travel together.

**The axis is stored-coefficient time**, and
:data:`~teb_vae.lag_attn_cfs.eval.lag_axis.GROUP_DELAY_CAVEAT` travels under every figure here and
in the emitted record: the coefficients come from a one-sided bank whose composed group delay is
the same order as the lag search, so an attention peak's position is not a physiological latency.

The lag heatmap needs the attention weights themselves, which are $427$ KiB per sample and
therefore retained only under ``eval_config.caps.attention``. A run that did not ask for them
emits no heatmap: the absence is silence rather than failure.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval.frames import (
    describe,
    finite_column,
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
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "attention"

#: What it writes.
PROFILE_FILENAME = "attention_profile.csv"
PER_HEAD_FILENAME = "attention_per_head.csv"
ENTROPY_FILENAME = "attention_entropy.csv"
PER_RECORDING_FILENAME = "attention_per_recording.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them.
PROFILE_FIGURE = "attention_profile.pdf"
HEATMAP_FIGURE = "lag_heatmap.pdf"

#: The retained tensor the heatmap needs, and the cap that decides whether it exists.
ATTENTION_TENSOR = "attn_weights"
ATTENTION_CAP = "attention"

#: The per-sample columns reduced per recording: the entropy and the ceiling it is read against,
#: so the ratio can be recomputed per recording rather than only pooled.
VALUE_COLUMNS: Tuple[str, ...] = (
    "attention_entropy_nats",
    "attention_entropy_attainable_nats",
)

#: The metrics resolved by cohort. Both, because a by-cohort entropy is only readable against the
#: ceiling that cohort's own anchors could reach: recordings differ in how many anchors they
#: score, so the attainable ceiling is a cohort property too.
GROUPED_METRICS: Tuple[str, ...] = VALUE_COLUMNS


#: How close the attainable ceiling must sit to $\log L$ to be reported as equal to it.
#:
#: The ceiling is a **float32** mean of $\log\min(t+1, L)$ over the scored anchors, chained through
#: two more means; $\log L$ is a float64 constant. At an untruncated geometry every term is
#: literally $\log L$ and the two still differ in the last bits of float32, which is a few parts in
#: $10^{8}$ -- so an exact test fails on arithmetic rather than on geometry.
#:
#: $10^{-6}$ discriminates with room to spare. The smallest truncation this could hide is one
#: anchor of the shipped $152$ seeing $L - 1$ lags instead of $L$, which moves the mean by
#: $(\log L - \log(L-1))/152 \approx 7 \times 10^{-5}$ -- two orders of magnitude above the bound.
CEILING_TOLERANCE = 1e-6


def truncated_anchor_accounting(geometry: Dict[str, Any], n_lags: int) -> Dict[str, Any]:
    r"""How many decoded anchors cannot see the whole lag window, from the model's own geometry.

    Lag $\ell$ at anchor $t$ refers to source step $t - \ell$, which does not exist for
    $\ell > t$, so the anchor's lag support is $\min(t + 1, L)$ and is complete only from
    $t = L - 1$ on. The decoded range is $[F, T_{\mathrm{valid}})$: nothing below the anchor floor
    is decoded at all, and the tail $H$ anchors have no fully observed future to score against.

    At this cell's shipped geometry -- $F = 133$, $T_{\mathrm{valid}} = 285$, $L = 91$ -- the count
    is **zero**, because the floor already exceeds $L - 1$. The number is derived here rather than
    written down, so a ``sweep_floor_*`` arm moves it instead of silently invalidating every
    ceiling read against it.

    Args:
        geometry: The collection record's geometry block.
        n_lags: Lag window width $L$.

    Returns:
        The decoded range, how many of its anchors are truncated, the first anchor with complete
        support, and the fraction. Zero counts on a geometry the record did not carry.
    """
    floor = int(geometry.get("anchor_floor", 0))
    t_valid = int(geometry.get("t_valid", 0))
    first_complete = max(int(n_lags) - 1, 0)
    n_decoded = max(t_valid - floor, 0)
    n_truncated = max(min(first_complete, t_valid) - floor, 0)
    return {
        "decoded_anchor_range": [floor, t_valid],
        "n_decoded_anchors": n_decoded,
        "first_untruncated_anchor": first_complete,
        "n_truncated_anchors": n_truncated,
        "truncated_fraction": (n_truncated / n_decoded) if n_decoded else float("nan"),
        "n_lags": int(n_lags),
    }


def measured_ceiling(
    entropies: List[Dict[str, Any]], truncation: Dict[str, Any], recorded: Dict[str, Any]
) -> Dict[str, Any]:
    r"""Compare the attainable entropy ceiling against $\log L$, and against what preflight said.

    Three independent statements about one property, and reporting fewer than all three is how a
    simplification outlives the geometry that justified it:

    * **Computed from the checkpoint's geometry.** Preflight's ``lag_support_margin_steps``
      $= \min_t \mathcal A - (L-1) - F_u$; non-negative means every scored anchor sees the whole
      lag window.
    * **Derived from the collection record's geometry.** The truncated-anchor count above, which is
      zero exactly when the same thing holds.
    * **Measured on the run's own numbers.** The attainable ceiling
      $\operatorname{mean}_t \log \min(t+1, L)$ against $\log L$.

    Nothing raises. A truncated-support run is a legitimate geometry and the restricted profile
    exists to handle it; what this records is whether the three agree, so a disagreement is a
    number rather than a reading nobody checked.

    Args:
        entropies: The entropy rows, for the measured attainable ceiling.
        truncation: The truncated-anchor accounting.
        recorded: What :func:`~teb_vae.lag_attn_cfs.eval.lag_axis.read_lag_support` returned.

    Returns:
        Both ceilings, their difference, whether they are equal to tolerance, and whether the three
        readings agree. ``None`` wherever a quantity was not measured, never a default.
    """
    pooled = next((row for row in entropies if row.get("scope") == "head_averaged"), {})
    attainable = float(pooled.get("ceiling_attainable_nats", float("nan")))
    log_l = float(pooled.get("ceiling_log_n_lags_nats", float("nan")))
    difference = (
        float("nan") if not (np.isfinite(attainable) and np.isfinite(log_l))
        else float(abs(attainable - log_l))
    )
    equal = None if not np.isfinite(difference) else bool(difference <= CEILING_TOLERANCE)
    untruncated_by_geometry = int(truncation.get("n_truncated_anchors") or 0) == 0
    expected = recorded.get("every_lag_valid_at_every_anchor")
    return {
        **recorded,
        "ceiling_attainable_nats": attainable,
        "ceiling_log_n_lags_nats": log_l,
        "ceiling_abs_difference": difference,
        "ceiling_equals_log_n_lags": equal,
        "ceiling_tolerance": CEILING_TOLERANCE,
        "untruncated_by_geometry_record": untruncated_by_geometry,
        # The three readings of one property. A mismatch means preflight, the geometry record and
        # the accumulated entropies describe different geometries, which no other number would show.
        "computed_and_observed_agree": (
            None if expected is None or equal is None
            else bool(expected == equal == untruncated_by_geometry)
        ),
    }


def entropy_rows(
    per_guid: pd.DataFrame, entropies: List[float], n_lags: int
) -> List[Dict[str, Any]]:
    r"""Report the attention entropy against both ceilings, pooled and per head.

    The two ceilings are $\log L$ -- what a lag window this wide could carry if every anchor saw
    all of it -- and the attainable one, $\operatorname{mean}_t \log \min(t + 1, L)$ over the
    anchors actually scored. They are named separately because the normalised entropy against the
    wrong one understates the model's spread by exactly the truncation.

    Args:
        per_guid: Per-recording means, for the measured entropy and its attainable ceiling.
        entropies: The per-head entropies from the pass's lag block, in head order.
        n_lags: Lag window width $L$, for the $\log L$ ceiling.

    Returns:
        One row for the head-averaged entropy and one per head, each with both ceilings and both
        normalised ratios.
    """
    ceiling_log_l = math.log(float(n_lags)) if n_lags > 0 else float("nan")
    # The mean over the recordings that measured one. An all-NaN column means no recording scored
    # an anchor, which is NaN rather than zero: a ceiling of zero would make every normalised
    # entropy infinite.
    attainable = finite_column(per_guid, "attention_entropy_attainable_nats")
    attainable = attainable[np.isfinite(attainable)]
    attainable_mean = float(attainable.mean()) if attainable.size else float("nan")
    rows: List[Dict[str, Any]] = []
    measured = finite_column(per_guid, "attention_entropy_nats")
    summary = describe(measured)
    rows.append(
        {
            "scope": "head_averaged",
            "head": None,
            "entropy_nats": summary["mean"],
            "n_recordings": summary["n"],
            **_ceilings(summary["mean"], ceiling_log_l, attainable_mean),
        }
    )
    for head, value in enumerate(entropies):
        rows.append(
            {
                "scope": "head",
                "head": head,
                "entropy_nats": float(value),
                "n_recordings": summary["n"],
                **_ceilings(float(value), ceiling_log_l, attainable_mean),
            }
        )
    return rows


def _ceilings(entropy: float, log_l: float, attainable: float) -> Dict[str, Any]:
    """The two ceilings and the two normalised entropies, under names that say which is which.

    Args:
        entropy: The measured entropy, in nats.
        log_l: $\\log L$, the ceiling the whole lag window would allow.
        attainable: The mean per-anchor attainable ceiling over the scored anchors.

    Returns:
        Both ceilings and both ratios. The attainable one is the one to read.
    """
    return {
        "ceiling_log_n_lags_nats": log_l,
        "ceiling_attainable_nats": attainable,
        "normalised_against_log_n_lags": (
            entropy / log_l if log_l and math.isfinite(log_l) and log_l > 0.0 else float("nan")
        ),
        "normalised_against_attainable": (
            entropy / attainable
            if attainable and math.isfinite(attainable) and attainable > 0.0
            else float("nan")
        ),
    }


def profile_frame(lag: Dict[str, Any], seconds: np.ndarray) -> pd.DataFrame:
    """Lay the three head-averaged profiles out as one table, keyed by lag.

    Args:
        lag: The pass's lag block.
        seconds: The compensated-seconds axis.

    Returns:
        One row per lag: its index, its compensated seconds, the raw head-averaged attention, the
        support-corrected form, and the form restricted to untruncated anchors.
    """
    raw = np.asarray(list(lag.get("attention_lag_profile") or []), dtype=np.float64)
    columns = ["lag_step", "compensated_seconds", "attention", "attention_support_corrected",
               "attention_untruncated"]
    if raw.size == 0:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(
        {
            "lag_step": np.arange(raw.size, dtype=int),
            "compensated_seconds": seconds[: raw.size],
            "attention": raw,
            "attention_support_corrected": padded_profile(
                np.asarray(
                    list(lag.get("attention_lag_profile_support_corrected") or []),
                    dtype=np.float64,
                ),
                raw.size,
            ),
            "attention_untruncated": padded_profile(
                np.asarray(
                    list(lag.get("attention_lag_profile_untruncated") or []), dtype=np.float64
                ),
                raw.size,
            ),
        }
    )


def per_head_frame(lag: Dict[str, Any], seconds: np.ndarray) -> pd.DataFrame:
    """Lay the per-head profiles out long, one row per head and lag.

    Long rather than wide so a head count change does not change the schema, and so a groupby on
    ``head`` is what reads one head's profile.

    Args:
        lag: The pass's lag block.
        seconds: The compensated-seconds axis.

    Returns:
        ``head``, ``lag_step``, ``compensated_seconds``, ``attention`` and the head's own KL.
        Empty with those columns when the pass carried no per-head profile.
    """
    profiles = list(lag.get("attention_lag_profile_per_head") or [])
    columns = ["head", "lag_step", "compensated_seconds", "attention", "head_kl_nats"]
    if not profiles:
        return pd.DataFrame(columns=columns)
    per_head_kl = list(lag.get("kld_per_head") or [])
    blocks: List[pd.DataFrame] = []
    for head, profile in enumerate(profiles):
        values = np.asarray(list(profile), dtype=np.float64)
        blocks.append(
            pd.DataFrame(
                {
                    "head": np.full(values.size, head, dtype=int),
                    "lag_step": np.arange(values.size, dtype=int),
                    "compensated_seconds": seconds[: values.size],
                    "attention": values,
                    "head_kl_nats": np.full(
                        values.size,
                        float(per_head_kl[head]) if head < len(per_head_kl) else np.nan,
                    ),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def build_profile_figure(
    profile: pd.DataFrame,
    per_head: pd.DataFrame,
    *,
    delay_steps: int,
    n_lags: int,
    truncation: Dict[str, Any],
) -> Any:
    """Draw the head-averaged profiles and the per-head panel, with the truncated region shaded.

    The shading is the point of the top panel: the lags that only the untruncated anchors could
    ever have contributed to are exactly where the restricted profile and the other two part
    company, and an unshaded version of this figure reads as a model that stops attending at long
    delays.

    Args:
        profile: The head-averaged profile table.
        per_head: The long per-head table.
        delay_steps: The causal input delay, for the axis.
        n_lags: Lag window width, so the axis spans the window even when the profile is empty.
        truncation: The truncated-anchor accounting, for the shaded span.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(2)
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    axis = axes[0, 0]
    figures.multi_line_panel(
        axis, seconds,
        np.vstack(
            [
                profile_column(profile, "attention", n_lags),
                profile_column(profile, "attention_support_corrected", n_lags),
                profile_column(profile, "attention_untruncated", n_lags),
            ]
        ),
        [
            "head-averaged (raw)",
            "head-averaged, support-corrected",
            "head-averaged, untruncated anchors only",
        ],
        title="Attention over lags",
        xlabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        ylabel="attention weight",
    )
    _shade_truncated(axis, seconds, truncation)

    heads = sorted(set(per_head["head"].tolist())) if len(per_head) else []
    curves = [
        padded_profile(
            np.asarray(per_head.loc[per_head["head"] == head, "attention"], dtype=np.float64),
            n_lags,
        )
        for head in heads
    ]
    figures.multi_line_panel(
        axes[1, 0], seconds,
        np.vstack(curves) if curves else np.zeros((0, n_lags)),
        [f"head {head}" for head in heads],
        title="Attention over lags, per head",
        xlabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        ylabel="attention weight",
    )
    _shade_truncated(axes[1, 0], seconds, truncation)
    figures.caveat_note(figure)
    return figure


def _shade_truncated(axis: Any, seconds: np.ndarray, truncation: Dict[str, Any]) -> None:
    """Shade the lags only the untruncated anchors could have contributed to.

    Structural rather than a finding, which is why it is drawn on both panels and labelled.

    Args:
        axis: Target axes, whose x-axis is the compensated-seconds lag axis.
        seconds: That axis's values.
        truncation: The truncated-anchor accounting.
    """
    n_truncated = int(truncation.get("n_truncated_anchors") or 0)
    if n_truncated <= 0 or seconds.size == 0:
        return
    floor = int((truncation.get("decoded_anchor_range") or [0, 0])[0])
    # A lag beyond the earliest decoded anchor's own index exists at *some* decoded anchors and
    # not at others; that band is where the truncation acts. At this cell's shipped floor there is
    # no such band and this function returns above -- an unshaded figure is then the honest one.
    first = min(max(floor, 0), seconds.size - 1)
    axis.axvspan(
        float(seconds[first]), float(seconds[-1]),
        color=figures.COLOR_LIGHT_GRAY, alpha=0.35, zorder=0,
        label=f"lags truncated at {n_truncated} of the decoded anchors",
    )
    axis.legend(fontsize=figures.FONT_SMALL, loc="best", ncol=2)


def build_heatmap_figure(
    attention: np.ndarray, *, row: int, delay_steps: int, geometry: Dict[str, Any]
) -> Any:
    r"""Draw one retained recording's attention as an anchor $\times$ lag field.

    Head-averaged, because the panel is about *when* rather than *which head*: the per-head
    resolution is in the profile figure, where four curves are readable and four heatmaps are not.

    ``interpolation='none'`` rather than the default resampling: this is a vector output whose
    reader is expected to index a cell, and a resampled cell boundary can land half a cell from
    where the data says it is.

    Args:
        attention: The retained weights, $(N, T, M, L)$.
        row: Which retained sample to draw.
        delay_steps: The causal input delay, for the lag axis.
        geometry: The collection record's geometry block, for the time axis.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(1)
    field = np.asarray(attention[row], dtype=np.float64)
    if field.ndim == 3:
        field = field.mean(axis=1)
    n_lags = int(field.shape[-1]) if field.size else 0
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    steps = int(field.shape[0]) if field.size else 0
    step_seconds = float(SECONDS_PER_STEP)
    # Row-reversed, and that is not cosmetic. The shared panel draws with ``origin='upper'``, so
    # data row 0 lands at the *top* of the extent -- which with an increasing seconds extent would
    # put lag $0$ at the largest label on the axis and silently invert the whole figure. Reversing
    # the rows puts lag 0 at the bottom and the lag increasing upward, which is the orientation
    # the training callback's lag panels already use.
    figures.heatmap_with_colorbar(
        figure, axes[0, 0], field.T[::-1],
        title=f"Attention by anchor and lag, retained sample row {row} (head-averaged)",
        xlabel="time in segment (s)",
        ylabel=figures.COEFFICIENT_LAG_AXIS_LABEL,
        symmetric=False,
        colorbar_label="attention weight",
        extent=(
            (0.0, steps * step_seconds, float(seconds[0]), float(seconds[-1]))
            if seconds.size
            else None
        ),
        interpolation="none",
    )
    # The anchor floor: nothing to its left is decoded at all on this cell, so the marker separates
    # a region with no forecast from one with a forecast rather than a warm-up from a trained range.
    floor = int(geometry.get("anchor_floor", 0))
    if floor > 0 and steps:
        axes[0, 0].axvline(
            floor * step_seconds, color=figures.COLOR_BLACK,
            linewidth=figures.LINE_THIN, linestyle="--",
        )
    figures.caveat_note(figure)
    return figure


def run_attention_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the attention per head, against its attainable ceiling, unbiased by truncation.

    Args:
        context: The analysis context, read for the pass's lag block, the per-sample table, the
            geometry record and the retained attention weights.
        eval_config: The validated block, read only to report which cap decided the retention.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the truncation accounting, the measured ceiling comparison, both
        entropies, the three argmaxes, the per-head KL, the group-delay caveat and the delay every
        reported lag was compensated by.
    """
    collection = context.collection
    per_sample = collection.per_sample
    results = dict(getattr(collection, "results", None) or {})
    record = dict(getattr(collection, "record", None) or {})
    geometry = dict(record.get("geometry") or {})
    lag = dict(results.get("lag") or {})
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    delay_steps = int(lag.get("delay_steps") or 0)
    n_lags = int(lag.get("n_lags") or len(lag.get("attention_lag_profile") or []))
    seconds = compensated_seconds_axis(n_lags, delay_steps)
    truncation = truncated_anchor_accounting(geometry, n_lags)

    profile = profile_frame(lag, seconds)
    profile.to_csv(directory / PROFILE_FILENAME, index=False)
    per_head = per_head_frame(lag, seconds)
    per_head.to_csv(directory / PER_HEAD_FILENAME, index=False)

    per_guid = per_recording_means(per_sample, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    entropies = entropy_rows(
        per_guid, [float(value) for value in lag.get("attention_entropy_per_head_nats") or []],
        n_lags,
    )
    pd.DataFrame(entropies).to_csv(directory / ENTROPY_FILENAME, index=False)
    # Read off the run's own preflight record rather than derived from three config keys, so an arm
    # that lowered the floor below the lag window is a number here rather than a stale assumption.
    ceiling = measured_ceiling(entropies, truncation, read_lag_support(output_dir))

    written: List[str] = [
        str(
            figures.render_to_pdf(
                build_profile_figure(
                    profile, per_head, delay_steps=delay_steps, n_lags=n_lags,
                    truncation=truncation,
                ),
                directory / PROFILE_FIGURE,
            ).name
        )
    ]
    heatmap = _emit_heatmap(collection, directory, delay_steps=delay_steps, geometry=geometry)
    if heatmap is not None:
        written.append(heatmap)

    return {
        "n_samples": scored_sample_count(per_sample, "attention_entropy_nats"),
        "composition": {"n_recordings": int(len(per_guid)), "num_heads": len(per_head_kl(lag))},
        "plan": {
            "capped": False,
            # Which cap decided whether the heatmap exists, named rather than implied by its
            # absence.
            "heatmap_cap": ATTENTION_CAP,
            "heatmap_cap_value": (eval_config.get("caps") or {}).get(ATTENTION_CAP, "absent"),
        },
        "delay_steps": delay_steps,
        "source_delay_is_max_over_channels": bool(
            lag.get("source_delay_is_max_over_channels", True)
        ),
        # The sentence every lag-resolved artifact carries. In the record as well as under both
        # figures, because ``summary.json`` is the artifact that gets quoted.
        "axis_caveat": GROUP_DELAY_CAVEAT,
        "truncation": truncation,
        # Measured, not assumed. Preflight's margin, the geometry record's truncated-anchor count
        # and the accumulated attainable ceiling against log L -- three readings of one property.
        "ceiling": ceiling,
        "entropy": entropies,
        # Three argmaxes: the raw one, the support-corrected one, and the one restricted to
        # anchors whose lag support is complete. Only the last is free of the renormalisation
        # bias, and it is the one an argmax claim should rest on.
        "argmax": {
            "raw_lag_step": lag.get("attention_argmax_lag_step"),
            "raw_compensated_seconds": lag.get("attention_lag_compensated_seconds"),
            "support_corrected_lag_step": lag.get(
                "attention_argmax_lag_step_support_corrected"
            ),
            "untruncated_lag_step": lag.get("attention_argmax_lag_step_untruncated"),
            "untruncated_compensated_seconds": lag.get(
                "attention_lag_compensated_seconds_untruncated"
            ),
            "restricted_to_anchors_from": truncation["first_untruncated_anchor"],
        },
        "kld_per_head": per_head_kl(lag),
        "kld_per_head_total_nats": lag.get("kld_per_head_total_nats"),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            PROFILE_FILENAME, PER_HEAD_FILENAME, ENTROPY_FILENAME, PER_RECORDING_FILENAME
        ] + written,
    }


def per_head_kl(lag: Dict[str, Any]) -> List[float]:
    """The per-head KL as plain floats, empty when the pass carried none.

    Args:
        lag: The pass's lag block.

    Returns:
        One value per head, in head order.
    """
    return [float(value) for value in (lag.get("kld_per_head") or [])]


def _emit_heatmap(
    collection: Any, directory: Path, *, delay_steps: int, geometry: Dict[str, Any]
) -> Optional[str]:
    """Draw the anchor-by-lag heatmap when the attention weights were retained.

    Retention is opt-in -- ``eval_config.caps.attention`` -- because the tensor is $(T, M, L)$,
    roughly $427$ KiB per sample and the largest single retention in the pass. A run that did not
    ask for it has not failed, so the absent figure is silence rather than an empty page.

    Args:
        collection: What the pass produced.
        directory: This analysis's output directory.
        delay_steps: The causal input delay, for the lag axis.
        geometry: The geometry block, for the warm-up marker.

    Returns:
        The filename written, or ``None``.
    """
    retained = dict(getattr(collection, "retained", None) or {})
    attention = retained.get(ATTENTION_TENSOR)
    if attention is None or len(attention) == 0:
        return None
    # Row 0 is a uniform draw rather than a prefix: the retention plan is already a seeded
    # stratified sample over the whole split.
    figure = build_heatmap_figure(
        np.asarray(attention), row=0, delay_steps=delay_steps, geometry=geometry
    )
    return str(figures.render_to_pdf(figure, directory / HEATMAP_FIGURE).name)
