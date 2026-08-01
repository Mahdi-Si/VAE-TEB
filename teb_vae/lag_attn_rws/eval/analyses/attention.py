r"""The attention itself: per head, against the entropy it can actually reach, unbiased by truncation.

Three things this reports that a head-averaged profile against $\log L$ cannot.

**Per head.** The posterior is head-structured: latent group $m$ is written by attention head $m$
alone, which is what makes the per-head KL an additive decomposition rather than an arbitrary
slice. Averaging the four heads before profiling discards exactly that -- four heads attending at
four different delays and one attending everywhere produce the same head-averaged profile. So the
per-head profiles, the per-head entropies and the per-head KL travel separately, and the
head-averaged profile is emitted beside them **named as such** rather than as "the" profile.

**Against the attainable ceiling.** A distribution over $n$ outcomes has entropy at most
$\log n$, and at anchor $t$ only $\min(t + 1, L)$ lags exist at all. At the shipped geometry the
trained anchors are $[30, 270)$ and $L = 91$, so $60$ of those $240$ anchors -- a quarter -- have
structurally truncated lag support and cannot reach $\log L$ however uniformly the model attends.
Measured against $\log L$, a model attending uniformly over everything available to it reads as
increasingly *concentrated* the earlier the anchor. Both ceilings are reported, distinctly named,
and the truncated-anchor count is recomputed from the model's own geometry rather than quoted.

**Unbiased by truncation.** The support correction divides each lag bin by its own
contributing-anchor count, which fixes the denominator. It cannot fix the numerator: attention
rows are normalised per anchor, so at a truncated anchor the mass that had no long lag to reach
was renormalised onto the short ones. Nothing in a per-lag count knows that happened. The
restricted profile -- over the anchors at which every lag exists -- is what removes it, and the
restricted argmax is the only one an argmax *claim* should rest on. It costs the truncated
anchors, so all three profiles travel together.

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

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval.frames import (
    describe,
    finite_column,
    grouped_frame_entry,
    per_recording_means,
    scored_sample_count,
)
from teb_vae.lag_attn_rws.eval.lag_axis import (
    compensated_seconds_axis,
    padded_profile,
    profile_column,
)
from teb_vae.lag_attn_rws.nets.lag_report import SECONDS_PER_STEP

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


def truncated_anchor_accounting(geometry: Dict[str, Any], n_lags: int) -> Dict[str, Any]:
    r"""How many trained anchors cannot see the whole lag window, from the model's own geometry.

    Lag $\ell$ at anchor $t$ refers to source step $t - \ell$, which does not exist for
    $\ell > t$, so the anchor's lag support is $\min(t + 1, L)$ and is complete only from
    $t = L - 1$ on. The trained range is $[w, T - H)$: the warm-up prefix carries no loss term and
    the tail $H$ anchors have no fully observed raw future.

    At the shipped geometry -- $w = 30$, $T - H = 270$, $L = 91$ -- that is $60$ of $240$
    anchors, and the number is derived here rather than written down, so a geometry change moves
    it instead of silently invalidating it.

    Args:
        geometry: The collection record's geometry block.
        n_lags: Lag window width $L$.

    Returns:
        The trained range, how many of its anchors are truncated, the first anchor with complete
        support, and the fraction. Zero counts on a geometry the record did not carry.
    """
    warmup = int(geometry.get("warmup", 0))
    t_valid = int(geometry.get("t_valid", 0))
    first_complete = max(int(n_lags) - 1, 0)
    n_trained = max(t_valid - warmup, 0)
    n_truncated = max(min(first_complete, t_valid) - warmup, 0)
    return {
        "trained_anchor_range": [warmup, t_valid],
        "n_trained_anchors": n_trained,
        "first_untruncated_anchor": first_complete,
        "n_truncated_anchors": n_truncated,
        "truncated_fraction": (n_truncated / n_trained) if n_trained else float("nan"),
        "n_lags": int(n_lags),
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
        xlabel=figures.COMPENSATED_LAG_AXIS_LABEL,
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
        xlabel=figures.COMPENSATED_LAG_AXIS_LABEL,
        ylabel="attention weight",
    )
    _shade_truncated(axes[1, 0], seconds, truncation)
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
    warmup = int((truncation.get("trained_anchor_range") or [0, 0])[0])
    # A lag beyond the earliest trained anchor's own index exists at *some* trained anchors and
    # not at others; that band is where the truncation acts.
    first = min(max(warmup, 0), seconds.size - 1)
    axis.axvspan(
        float(seconds[first]), float(seconds[-1]),
        color=figures.COLOR_LIGHT_GRAY, alpha=0.35, zorder=0,
        label=f"lags truncated at {n_truncated} of the trained anchors",
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
        ylabel=figures.COMPENSATED_LAG_AXIS_LABEL,
        symmetric=False,
        colorbar_label="attention weight",
        extent=(
            (0.0, steps * step_seconds, float(seconds[0]), float(seconds[-1]))
            if seconds.size
            else None
        ),
        interpolation="none",
    )
    warmup = int(geometry.get("warmup", 0))
    if warmup > 0 and steps:
        axes[0, 0].axvline(
            warmup * step_seconds, color=figures.COLOR_BLACK, linewidth=figures.LINE_THIN, linestyle="--"
        )
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
        The protocol's keys plus the truncation accounting, both entropies, the three argmaxes,
        the per-head KL and the delay every reported lag was compensated by.
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
        "truncation": truncation,
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
