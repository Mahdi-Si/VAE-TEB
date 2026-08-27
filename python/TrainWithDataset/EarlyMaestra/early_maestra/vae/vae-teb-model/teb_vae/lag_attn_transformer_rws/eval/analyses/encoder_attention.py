r"""What the encoder self-attention attends to -- the one question this architecture makes askable.

The ``attention`` analysis beside this one profiles the **lag cross-attention**: $M = 4$ heads over
$L = 91$ lags, the mechanism that decides which past *source* neighbourhood informs the latent.
This profiles a different mechanism entirely -- $H_e$ heads over the within-stream time axis, a
full causal prefix on the target and a $W_U$-step window on the source -- and that mechanism is the
whole content of the encoder replacement. Nothing else in the repository reads it.

Three readouts, each per block, per stream, and cut by clinical class on the shared cohort grid:

1. **Per-head entropy against its truncation-aware ceiling.** $\operatorname{mean}_t \log \min(t+1,
   c)$, with $c = T$ for the target and $c = W_U$ for the source -- never $\log T$, for the same
   reason the lag analysis refuses $\log L$: at anchor $t$ only $\min(t+1, c)$ keys exist at all,
   so a head attending uniformly over everything available to it reads as increasingly
   *concentrated* the earlier the anchor when measured against a ceiling it could not reach.
2. **Attention mass by temporal distance** $t - j$. This tests the design claim directly: that the
   target encoder gives the prior content-dependent access to long-range history the recurrent
   branch could not, and that the source encoder stays inside its window.
3. **The measured source reach against the lag range.** The structural bound is
   $R_U = R_{\mathrm{conv}} + N_U(W_U - 1)$, deliberately shorter than the lag search, because an
   encoder whose reach exceeded the lag range would already be doing the alignment the lag
   cross-attention exists to do. The mass-weighted quantiles measure what it *uses*, which gives
   the whole ``sweep_window_*`` arm family a measured x-axis instead of a configured one.

**It computes no test and produces no verdict.** Like ``distributions``, it describes a mechanism
rather than adjudicating a difference: a separation visible here is a reason to look rather than a
finding. It does register headline scalars, and it has to -- the arm tables read the headline block
and nothing else, so a measured reach that stayed inside a CSV could never reach the table that
needs it. No threshold is registered beside them, because there is none anyone has earned the right
to set.

**It runs its own bounded pass**, off the task and the loader on the context rather than off the
collection tables, which is the precedent ``samples`` already sets. The quantity it needs is a
$(B, H_e, T, T)$ probability tensor the collection pass does not produce and could not retain --
roughly $46$ MB per batch at the evaluation batch size -- so threading it through the shared
retention would edit a shared module for one model's benefit and pay for it on every run. The pass
is therefore opt-in: ``eval_config.caps.encoder_attention`` caps how many segments it scores, and
an absent cap means zero, per this pipeline's opt-in rule. A run that did not ask for it records a
skip naming the key, and costs nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn_rws.eval import figures_seam as figures
from teb_vae.lag_attn_rws.eval._reuse import labels
from teb_vae.lag_attn_rws.eval.cohort import ordered_groups
from teb_vae.lag_attn_rws.eval.frames import grouped_frame_entry, per_recording_means
from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_transformer_rws.eval.encoder_attention import (
    POOLED_CLASS,
    REACH_QUANTILES,
    STREAMS,
    composed_reach,
    mass_quantile,
    run_encoder_attention_pass,
    stratified_segment_loader,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "encoder_attention"

#: The cap that decides whether this analysis runs at all, and how much of the split it sees.
CAP_NAME = "encoder_attention"

#: What it writes.
ENTROPY_FILENAME = "encoder_attention_entropy.csv"
DISTANCE_FILENAME = "encoder_attention_distance.csv"
REACH_FILENAME = "encoder_attention_reach.csv"
PER_RECORDING_FILENAME = "encoder_attention_per_recording.csv"

#: The figures, named as ``FIGURE_GUIDE.md`` names them. The two class-resolved ones are drawn
#: here rather than by the runner's violin fan-out: that fan-out resolves per-*recording* scalars,
#: and these two are per-head and per-distance fields, which a violin cannot carry. The fan-out
#: still runs, over :data:`PER_RECORDING_FILENAME`, and writes its own files beside these.
ENTROPY_FIGURE = "encoder_attention_entropy"
ENTROPY_CLASS_FIGURE = "encoder_attention_entropy_by_clinical_class"
DISTANCE_FIGURE = "encoder_attention_distance"
DISTANCE_CLASS_FIGURE = "encoder_attention_distance_by_clinical_class"
HEATMAP_FIGURE = "encoder_attention_heatmap"

#: The per-segment columns reduced per recording, and the ones the grouped fan-out resolves by
#: cohort. Both streams' entropy ratio and mean distance: a class contrast on one stream alone
#: cannot say whether a difference is the encoder's or the cohort's.
VALUE_COLUMNS: Tuple[str, ...] = (
    "encoder_entropy_ratio_target",
    "encoder_entropy_ratio_source",
    "encoder_attention_distance_target_steps",
    "encoder_attention_distance_source_steps",
)

#: The headline scalars this analysis registers, as the binding carries them: the two per-stream
#: entropy ratios, and the source reach's two quantiles in steps and in seconds. Small and
#: deliberate -- everything else here is a field rather than a number, and a field flattened into
#: one scalar is the misreading the per-head tables exist to prevent.
HEADLINE_KEYS: Tuple[str, ...] = (
    "entropy_ratio_target",
    "entropy_ratio_source",
    "source_reach_median_steps",
    "source_reach_median_seconds",
    "source_reach_p95_steps",
    "source_reach_p95_seconds",
)


# =============================================================================
# Cohort order
# =============================================================================
def cohort_order(per_segment: pd.DataFrame) -> List[str]:
    """Return the cohorts every table and figure is written in, pooled first.

    Args:
        per_segment: The per-segment frame, read for the classes actually present.

    Returns:
        :data:`POOLED_CLASS`, then the clinical classes in the evaluation's one cohort order --
        HIE, acidosis, healthy, worst first -- rather than alphabetically, which would put
        ``acidosis`` first on every figure.
    """
    if per_segment.empty or labels.CLASS_COLUMN not in per_segment.columns:
        return [POOLED_CLASS]
    present = labels.distinct_groups(list(per_segment[labels.CLASS_COLUMN]))
    return [POOLED_CLASS] + list(ordered_groups(present, labels.CLASS_COLUMN))


# =============================================================================
# The tables
# =============================================================================
def entropy_frame(result: Any, cohorts: Sequence[str]) -> pd.DataFrame:
    r"""Lay the per-head entropies out long, one row per cohort, stream, block and head.

    Long rather than wide so a head- or block-count change does not change the schema, and so
    reading one head's line is a ``groupby``.

    Args:
        result: The pass's result.
        cohorts: The cohorts to emit, in output order.

    Returns:
        The measured entropy, the attainable ceiling it is read against, their ratio, and the
        denominators behind both.
    """
    rows: List[Dict[str, Any]] = []
    for cohort in cohorts:
        for stream in STREAMS:
            for (name, block_stream, block), accumulator in sorted(result.accumulators.items()):
                if name != cohort or block_stream != stream:
                    continue
                entropy = accumulator.entropy_mean
                ratio = accumulator.ratio_mean
                for head in range(accumulator.n_heads):
                    rows.append(
                        {
                            labels.CLASS_COLUMN: cohort,
                            "stream": stream,
                            "block": int(block),
                            "head": int(head),
                            "entropy_nats": float(entropy[head]),
                            "ceiling_attainable_nats": accumulator.ceiling_mean,
                            "entropy_ratio": float(ratio[head]),
                            "n_anchors": int(accumulator.n_anchors),
                            "n_segments": int(accumulator.n_segments),
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=[
            labels.CLASS_COLUMN, "stream", "block", "head", "entropy_nats",
            "ceiling_attainable_nats", "entropy_ratio", "n_anchors", "n_segments",
        ],
    )


def distance_frame(result: Any, cohorts: Sequence[str]) -> pd.DataFrame:
    r"""Lay the mass-by-distance profiles out long, truncated at each block's own reach.

    A source block admits no key beyond $W - 1$, so every bin past it is exactly zero by
    construction; those bins are dropped rather than written out, because a table two thirds
    zeroes invites a reader to look for the shape in the zeroes. That the zeroes are exact is
    asserted in the test suite against the accumulator itself, which is where a claim about the
    mask belongs.

    ``mass`` is divided by **every** scored anchor, while a bin at distance $d$ can only be written
    by the anchors $t \ge d$. So a far bin is averaged over anchors that could not reach it and
    reads low for that reason alone -- the same anchor-availability bias the shared package's lag
    profiles carry, and it is reported the same way: ``n_contributing_anchors`` is the count that
    could write each bin, so a reader who wants the availability-corrected profile multiplies by
    ``n_anchors / n_contributing_anchors`` rather than inferring a shape from the falloff. The
    uncorrected form stays the emitted one because it is what the quantiles, the mean distance and
    the composed reach are all taken over, and a table whose column disagreed with them would be
    the worse trap.

    Args:
        result: The pass's result.
        cohorts: The cohorts to emit, in output order.

    Returns:
        One row per cohort, stream, block, head and distance bin.
    """
    rows: List[Dict[str, Any]] = []
    start, stop = int(result.anchor_range[0]), int(result.anchor_range[1])
    for cohort in cohorts:
        for stream in STREAMS:
            window = result.geometry[stream]["attention_window"]
            reach = result.seq_len if window is None else min(int(window), result.seq_len)
            for (name, block_stream, block), accumulator in sorted(result.accumulators.items()):
                if name != cohort or block_stream != stream:
                    continue
                profile = accumulator.distance_profile
                for head in range(accumulator.n_heads):
                    for distance in range(reach):
                        rows.append(
                            {
                                labels.CLASS_COLUMN: cohort,
                                "stream": stream,
                                "block": int(block),
                                "head": int(head),
                                "distance_steps": int(distance),
                                "distance_seconds": float(distance * SECONDS_PER_STEP),
                                "mass": float(profile[head, distance]),
                                "n_contributing_anchors": max(stop - max(start, distance), 0),
                                "n_anchors": max(stop - start, 0),
                            }
                        )
    return pd.DataFrame(
        rows,
        columns=[
            labels.CLASS_COLUMN, "stream", "block", "head", "distance_steps",
            "distance_seconds", "mass", "n_contributing_anchors", "n_anchors",
        ],
    )


def reach_frame(result: Any, cohorts: Sequence[str], *, lag_range_steps: int) -> pd.DataFrame:
    r"""Compose the per-block measured distances into each stream's reach, against its bounds.

    Every row carries both the block's own quantiles and the whole stack's composed reach, so a
    reader groups by ``(clinical_class, stream)`` and reads either without joining a second table.

    Args:
        result: The pass's result.
        cohorts: The cohorts to emit, in output order.
        lag_range_steps: The furthest lag the cross-attention searches, which is what the source
            reach is meant to stay inside.

    Returns:
        One row per cohort, stream and block. The columns are declared rather than inferred, as
        the two frames above declare theirs: a pass that scored no segments must still write a
        table with a schema, because the headline block reads this frame by column name and a
        column-less frame would turn an empty measurement into a ``KeyError`` where the protocol
        asks for six nulls.
    """
    rows: List[Dict[str, Any]] = []
    for cohort in cohorts:
        for stream in STREAMS:
            geometry = result.geometry[stream]
            blocks = sorted(
                (
                    (block, accumulator)
                    for (name, block_stream, block), accumulator in result.accumulators.items()
                    if name == cohort and block_stream == stream
                ),
                key=lambda item: item[0],
            )
            if not blocks:
                continue
            quantiles = {
                quantile: [
                    mass_quantile(accumulator.distance_profile.mean(axis=0), quantile)
                    for _block, accumulator in blocks
                ]
                for quantile in REACH_QUANTILES
            }
            composed = {
                quantile: composed_reach(
                    geometry["conv_reach_steps"], values, sequence_length=result.seq_len
                )
                for quantile, values in quantiles.items()
            }
            bound = geometry["structural_bound_steps"]
            for position, (block, accumulator) in enumerate(blocks):
                rows.append(
                    {
                        labels.CLASS_COLUMN: cohort,
                        "stream": stream,
                        "block": int(block),
                        "median_distance_steps": quantiles[0.5][position],
                        "median_distance_seconds": quantiles[0.5][position] * SECONDS_PER_STEP,
                        "p95_distance_steps": quantiles[0.95][position],
                        "p95_distance_seconds": quantiles[0.95][position] * SECONDS_PER_STEP,
                        "conv_reach_steps": geometry["conv_reach_steps"],
                        "n_attention_blocks": geometry["n_attention_blocks"],
                        "attention_window": geometry["attention_window"],
                        "composed_reach_median_steps": composed[0.5],
                        "composed_reach_median_seconds": composed[0.5] * SECONDS_PER_STEP,
                        "composed_reach_p95_steps": composed[0.95],
                        "composed_reach_p95_seconds": composed[0.95] * SECONDS_PER_STEP,
                        "structural_bound_steps": bound,
                        "structural_bound_seconds": geometry["structural_bound_seconds"],
                        "structural_bound_absent": bool(geometry["structural_bound_absent"]),
                        "lag_range_max_steps": int(lag_range_steps),
                        "lag_range_max_seconds": int(lag_range_steps) * SECONDS_PER_STEP,
                        # Against the *measured* reach rather than the structural one, which is
                        # the whole point: the bound says what the encoder may do, this says what
                        # it did. NaN-safe -- an unmeasured reach is neither inside nor outside.
                        "measured_reach_inside_lag_range": (
                            bool(composed[0.95] < float(lag_range_steps))
                            if np.isfinite(composed[0.95]) else None
                        ),
                        "n_segments": int(accumulator.n_segments),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            labels.CLASS_COLUMN, "stream", "block",
            "median_distance_steps", "median_distance_seconds",
            "p95_distance_steps", "p95_distance_seconds",
            "conv_reach_steps", "n_attention_blocks", "attention_window",
            "composed_reach_median_steps", "composed_reach_median_seconds",
            "composed_reach_p95_steps", "composed_reach_p95_seconds",
            "structural_bound_steps", "structural_bound_seconds", "structural_bound_absent",
            "lag_range_max_steps", "lag_range_max_seconds",
            "measured_reach_inside_lag_range", "n_segments",
        ],
    )


# =============================================================================
# The figures
# =============================================================================
def build_entropy_figure(entropy: pd.DataFrame) -> Any:
    """Draw the per-head entropy ratio, one panel per stream, bars grouped by block.

    The ceiling is drawn as a line at $1$ rather than left implicit: the ratio is against the
    entropy that block could actually reach, so $1$ means "uniform over everything admitted" and a
    figure without that line invites a reader to pick their own reference.

    Args:
        entropy: The entropy table, pooled rows only.

    Returns:
        The figure; the caller renders and closes it.
    """
    figure, axes = figures.new_figure(len(STREAMS))
    for position, stream in enumerate(STREAMS):
        axis = axes[position, 0]
        rows = entropy[entropy["stream"] == stream].sort_values(["block", "head"])
        values = np.asarray(rows["entropy_ratio"], dtype=np.float64)
        if values.size and np.isfinite(values).any():
            axis.bar(
                np.arange(values.size, dtype=np.float64), values,
                color=figures.COLOR_BLUE, alpha=0.85, width=0.8,
            )
            axis.axhline(
                1.0, color=figures.COLOR_VERMILLION, linestyle="--",
                linewidth=figures.LINE_REGULAR,
                label="uniform over every admitted key",
            )
            axis.set_xticks(np.arange(values.size, dtype=np.float64))
            axis.set_xticklabels(
                [f"b{int(block)}h{int(head)}" for block, head in zip(rows["block"], rows["head"])],
                fontsize=figures.FONT_TINY,
            )
            axis.legend(fontsize=figures.FONT_LABEL, loc="best")
        else:
            axis.text(
                0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
            )
        axis.set_title(f"{stream} encoder: attention entropy against its attainable ceiling")
        axis.set_xlabel("block and head")
        axis.set_ylabel("entropy / attainable ceiling")
        figures.style_axes(axis)
    return figure


def build_entropy_class_figure(entropy: pd.DataFrame, cohorts: Sequence[str]) -> Any:
    """Draw the block- and head-averaged entropy ratio per clinical class, one panel per stream.

    Args:
        entropy: The entropy table, all cohorts.
        cohorts: The cohorts in clinical order, pooled first.

    Returns:
        The figure; the caller renders and closes it.
    """
    classes = [cohort for cohort in cohorts if cohort != POOLED_CLASS]
    palette = figures.group_colors(classes)
    figure, axes = figures.new_figure(len(STREAMS))
    for position, stream in enumerate(STREAMS):
        axis = axes[position, 0]
        rows = entropy[entropy["stream"] == stream]
        values = [
            float(np.nanmean(rows.loc[rows[labels.CLASS_COLUMN] == name, "entropy_ratio"]))
            if (rows[labels.CLASS_COLUMN] == name).any() else np.nan
            for name in classes
        ]
        pooled = rows.loc[rows[labels.CLASS_COLUMN] == POOLED_CLASS, "entropy_ratio"]
        if classes and np.isfinite(np.asarray(values, dtype=np.float64)).any():
            axis.bar(
                np.arange(len(classes), dtype=np.float64), values,
                color=[palette.get(name, figures.COLOR_GRAY) for name in classes],
                alpha=0.9, width=0.7,
            )
            axis.set_xticks(np.arange(len(classes), dtype=np.float64))
            axis.set_xticklabels(classes, fontsize=figures.FONT_SMALL)
            if len(pooled):
                axis.axhline(
                    float(np.nanmean(pooled)), color=figures.COLOR_BLACK, linestyle="--",
                    linewidth=figures.LINE_THIN, label="pooled",
                )
                axis.legend(fontsize=figures.FONT_LABEL, loc="best")
        else:
            axis.text(
                0.5, 0.5, figures.EMPTY_NOTE, ha="center", va="center", transform=axis.transAxes
            )
        axis.set_title(f"{stream} encoder: entropy ratio by clinical class")
        axis.set_xlabel("clinical class, in ascending severity")
        axis.set_ylabel("entropy / attainable ceiling")
        figures.style_axes(axis)
    return figure


def build_distance_figure(distance: pd.DataFrame, geometry: Dict[str, Dict[str, Any]]) -> Any:
    r"""Draw the mass-by-distance profile per head, one panel per block, window bound marked.

    Args:
        distance: The distance table, pooled rows only.
        geometry: Per-stream structural facts, for the window marker.

    Returns:
        The figure; the caller renders and closes it.
    """
    panels = [
        (stream, int(block))
        for stream in STREAMS
        for block in sorted(set(distance.loc[distance["stream"] == stream, "block"]))
    ]
    figure, axes = figures.new_figure(max(len(panels), 1))
    for position, (stream, block) in enumerate(panels):
        axis = axes[position, 0]
        rows = distance[(distance["stream"] == stream) & (distance["block"] == block)]
        heads = sorted(set(int(head) for head in rows["head"]))
        curves = [
            np.asarray(rows.loc[rows["head"] == head, "mass"], dtype=np.float64) for head in heads
        ]
        seconds = np.asarray(
            rows.loc[rows["head"] == heads[0], "distance_seconds"], dtype=np.float64
        ) if heads else np.zeros(0)
        figures.multi_line_panel(
            axis, seconds,
            np.vstack(curves) if curves else np.zeros((0, seconds.size)),
            [f"head {head}" for head in heads],
            title=f"{stream} encoder, block {block}: attention mass by temporal distance",
            xlabel="temporal distance t - j (s)",
            ylabel="attention mass",
        )
        window = geometry[stream]["attention_window"]
        if window is not None:
            axis.axvline(
                float((int(window) - 1) * SECONDS_PER_STEP), color=figures.COLOR_BLACK,
                linestyle="--", linewidth=figures.LINE_THIN,
            )
    return figure


def build_distance_class_figure(
    distance: pd.DataFrame, cohorts: Sequence[str], geometry: Dict[str, Dict[str, Any]]
) -> Any:
    """Draw the block- and head-averaged distance profile per clinical class, one panel per stream.

    Args:
        distance: The distance table, all cohorts.
        cohorts: The cohorts in clinical order, pooled first.
        geometry: Per-stream structural facts, for the window marker.

    Returns:
        The figure; the caller renders and closes it.
    """
    classes = [cohort for cohort in cohorts if cohort != POOLED_CLASS]
    palette = figures.group_colors(classes)
    figure, axes = figures.new_figure(len(STREAMS))
    for position, stream in enumerate(STREAMS):
        axis = axes[position, 0]
        rows = distance[distance["stream"] == stream]
        bins = sorted(set(int(value) for value in rows["distance_steps"]))
        seconds = np.asarray(bins, dtype=np.float64) * SECONDS_PER_STEP
        curves = []
        drawn = []
        for name in classes:
            block = rows[rows[labels.CLASS_COLUMN] == name]
            if not len(block):
                continue
            profile = block.groupby("distance_steps")["mass"].mean().reindex(bins)
            curves.append(np.asarray(profile, dtype=np.float64))
            drawn.append(name)
        figures.multi_line_panel(
            axis, seconds,
            np.vstack(curves) if curves else np.zeros((0, seconds.size)),
            list(drawn),
            title=f"{stream} encoder: attention mass by temporal distance, by clinical class",
            xlabel="temporal distance t - j (s)",
            ylabel="attention mass",
        )
        # Recoloured by label rather than by position: the shared panel skips a row that is
        # entirely non-finite, so a cohort that measured nothing would shift every colour after it
        # onto the wrong class. The legend is rebuilt afterwards so its swatches follow.
        for line in axis.get_lines():
            colour = palette.get(str(line.get_label()))
            if colour is not None:
                line.set_color(colour)
        if drawn:
            axis.legend(fontsize=figures.FONT_LABEL, loc="best", ncol=2)
        window = geometry[stream]["attention_window"]
        if window is not None:
            axis.axvline(
                float((int(window) - 1) * SECONDS_PER_STEP), color=figures.COLOR_BLACK,
                linestyle="--", linewidth=figures.LINE_THIN,
            )
    return figure


def build_heatmap_figure(result: Any) -> Any:
    r"""Draw one segment's attention map per block, anchor $\times$ key, head-averaged.

    ``interpolation='none'`` rather than the default resampling, as the sibling's page builder
    does: this is a vector output whose reader is expected to index a cell, and a resampled cell
    boundary can land half a cell from where the data says it is.

    Head-averaged because the panel is about *when* rather than *which head*: the per-head
    resolution is in the profile figure, where four curves are readable and four heatmaps are not.

    Args:
        result: The pass's result, read for the retained maps.

    Returns:
        The figure; the caller renders and closes it.
    """
    panels = sorted(result.heatmaps)
    figure, axes = figures.new_figure(max(len(panels), 1))
    span = float(result.seq_len * SECONDS_PER_STEP)
    for position, key in enumerate(panels):
        stream, block = key
        figures.heatmap_with_colorbar(
            figure, axes[position, 0], np.asarray(result.heatmaps[key], dtype=np.float64),
            title=f"{stream} encoder, block {block}: one segment's attention, head-averaged",
            xlabel="key step j (s)",
            ylabel="anchor step t (s)",
            symmetric=False,
            colorbar_label="attention weight",
            extent=(0.0, span, span, 0.0),
            interpolation="none",
        )
    return figure


# =============================================================================
# The registry entry point
# =============================================================================
def run_encoder_attention_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Profile both encoders' self-attention over a capped, stratified draw of segments.

    Args:
        context: The analysis context, read -- uniquely among this package's analyses, and for the
            same reason ``samples`` is -- for the task and the loader the pass runs on.
        eval_config: The validated block, for ``caps.encoder_attention`` and the draw's seed.
        output_dir: The results directory; this analysis writes into its own subdirectory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the headline block, the draw record and the structural geometry
        every measured number is read against. A pass with no cap, or with no model, records a
        skip -- with the headline keys present and ``null``, so an arm table's column exists
        whether the analysis ran or not.
    """
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    cap = (eval_config.get("caps") or {}).get(CAP_NAME)
    task, loader = getattr(context, "task", None), getattr(context, "loader", None)
    if cap is None:
        return _skipped(
            f"retention is opt-in in this pipeline and eval_config.caps.{CAP_NAME} is absent, so "
            f"this analysis scores no segments; set it to the number of segments the encoder "
            f"attention should be profiled over. The pass recomputes a (B, H_e, T, T) probability "
            f"tensor the model never materialises, which is why it is asked for rather than "
            f"assumed."
        )
    if task is None or loader is None:
        return _skipped(
            f"the encoder attention is recomputed from the model's own parameters over its own "
            f"bounded pass, so it needs a model and a loader; this pass built neither, which is "
            f"what an offline re-run against a finished directory is",
            cap=int(cap),
        )

    seed = int(eval_config.get("seed", 0))
    segments, draw = stratified_segment_loader(loader, cap=int(cap), seed=seed)
    result = run_encoder_attention_pass(task, segments)

    per_segment = pd.DataFrame(result.per_segment)
    cohorts = cohort_order(per_segment)
    lag_range_steps = int(getattr(task.orig_model, "max_lag", 0))

    entropy = entropy_frame(result, cohorts)
    entropy.to_csv(directory / ENTROPY_FILENAME, index=False)
    distance = distance_frame(result, cohorts)
    distance.to_csv(directory / DISTANCE_FILENAME, index=False)
    reach = reach_frame(result, cohorts, lag_range_steps=lag_range_steps)
    reach.to_csv(directory / REACH_FILENAME, index=False)
    per_guid = per_recording_means(per_segment, VALUE_COLUMNS)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)

    pooled_entropy = entropy[entropy[labels.CLASS_COLUMN] == POOLED_CLASS]
    pooled_distance = distance[distance[labels.CLASS_COLUMN] == POOLED_CLASS]
    written = [
        str(figures.render_figure(
            build_entropy_figure(pooled_entropy), directory / ENTROPY_FIGURE
        ).name),
        str(figures.render_figure(
            build_entropy_class_figure(entropy, cohorts), directory / ENTROPY_CLASS_FIGURE
        ).name),
        str(figures.render_figure(
            build_distance_figure(pooled_distance, result.geometry), directory / DISTANCE_FIGURE
        ).name),
        str(figures.render_figure(
            build_distance_class_figure(distance, cohorts, result.geometry),
            directory / DISTANCE_CLASS_FIGURE,
        ).name),
        str(figures.render_figure(build_heatmap_figure(result), directory / HEATMAP_FIGURE).name),
    ]

    logger.info(
        f"{ANALYSIS_DIRNAME}: profiled {result.n_segments} segment(s) over "
        f"{len(result.accumulators)} cohort-block cell(s), anchors "
        f"{result.anchor_range[0]}..{result.anchor_range[1]}"
    )
    return {
        "n_samples": int(result.n_segments),
        "composition": {
            "n_recordings": int(len(per_guid)),
            "n_shards_reached": draw["n_shards_drawn"],
            "n_per_class": {
                cohort: int(
                    (per_segment[labels.CLASS_COLUMN] == cohort).sum()
                    if labels.CLASS_COLUMN in per_segment.columns else 0
                )
                for cohort in cohorts
                if cohort != POOLED_CLASS
            },
        },
        # Capped by construction, so the coverage block compares this analysis's population
        # against nobody else's -- it deliberately scores a draw rather than the split.
        "plan": {"capped": True, "cap": int(cap), "cap_key": CAP_NAME, "seed": seed, **draw},
        "anchor_range": list(result.anchor_range),
        "n_heads": int(result.n_heads),
        # The batch's own $T$, not the constructor's: the mask, the ceiling and the distance axis
        # are all sliced to it, so a shard shorter than the geometry the model was built for makes
        # every number here describe that shorter sequence and the record has to say so.
        "sequence_length": int(result.seq_len),
        "geometry": result.geometry,
        "lag_range_max_steps": lag_range_steps,
        "lag_range_max_seconds": lag_range_steps * SECONDS_PER_STEP,
        "headline": headline_block(entropy, reach),
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, VALUE_COLUMNS)
        ],
        "files": [
            ENTROPY_FILENAME, DISTANCE_FILENAME, REACH_FILENAME, PER_RECORDING_FILENAME
        ] + written,
    }


def headline_block(entropy: pd.DataFrame, reach: pd.DataFrame) -> Dict[str, Any]:
    """Flatten the six registered scalars out of the pooled rows of the two tables.

    Args:
        entropy: The entropy table.
        reach: The reach table.

    Returns:
        Every key of :data:`HEADLINE_KEYS`, ``None`` where the pass measured nothing -- never
        omitted, so the arm table's column exists either way.
    """
    block: Dict[str, Any] = {name: None for name in HEADLINE_KEYS}
    pooled_entropy = entropy[entropy[labels.CLASS_COLUMN] == POOLED_CLASS]
    for stream in STREAMS:
        values = np.asarray(
            pooled_entropy.loc[pooled_entropy["stream"] == stream, "entropy_ratio"],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        block[f"entropy_ratio_{stream}"] = float(finite.mean()) if finite.size else None

    pooled_reach = reach[
        (reach[labels.CLASS_COLUMN] == POOLED_CLASS) & (reach["stream"] == "source")
    ]
    if len(pooled_reach):
        row = pooled_reach.iloc[0]
        for name, column in (
            ("source_reach_median_steps", "composed_reach_median_steps"),
            ("source_reach_median_seconds", "composed_reach_median_seconds"),
            ("source_reach_p95_steps", "composed_reach_p95_steps"),
            ("source_reach_p95_seconds", "composed_reach_p95_seconds"),
        ):
            value = float(row[column])
            block[name] = value if np.isfinite(value) else None
    return block


def _skipped(reason: str, *, cap: Optional[int] = None) -> Dict[str, Any]:
    """Return the protocol's keys for a pass that scored nothing, saying why.

    Args:
        reason: What was missing, in the package's ``skipped`` / ``reason`` shape.
        cap: The cap that was set, when one was.

    Returns:
        The skip record, with every headline key present and ``null``.
    """
    logger.warning(f"{ANALYSIS_DIRNAME}: skipped -- {reason}")
    return {
        "n_samples": None,
        "composition": {},
        "plan": {"capped": True, "cap": cap, "cap_key": CAP_NAME},
        "skipped": True,
        "reason": reason,
        "headline": {name: None for name in HEADLINE_KEYS},
        "files": [],
    }
