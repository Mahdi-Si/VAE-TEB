r"""How the posterior event gate behaves: when it is shut, who shut it, and what an open gate buys.

The posterior applies the effective gate $g^{\mathrm{eff}}_t = g_t\,o_t$ to both of its residuals,
where $g_t$ is the learned gate and $o_t$ says whether at least one UA lag key survived the
signal-quality mask. The objective charges $\lambda\,\mathbb E_t[g^{\mathrm{eff}}_t]$, so a gate
that stays open is a gate the model is paying for. The mean of $g^{\mathrm{eff}}_t$ alone cannot
say whether that payment is used well, because $g^{\mathrm{eff}}_t = 0$ for two different reasons:

* **forced shut**, $o_t = 0$ -- the data removed every key, and $q = p$ whatever the model wants;
* **model shut**, $o_t = 1$ and $g_t < g_{\mathrm{shut}}$ -- the UA was there and the model chose
  not to read it.

So this analysis assigns every scored anchor to one of four states
(:func:`~teb_vae.lag_attn_cfs.gate_states.gate_states`, the rule the sample page uses too) and
reports, per state, the share of anchors, the source-conditioned KL $K_t$ and the forecast gain.
It then reads the learned gate against the two clocks a gate should respond to: the time before
delivery and the time since the last contraction onset. A gate that tracks contractions opens
shortly after an onset; a gate that only tracks the UA mask is constant wherever the keys are
open.

**Every summary is a mean over recordings**, after a mean within each recording, so a long
recording does not outvote a short one. The two time profiles are anchor-level means per bin,
because one recording rarely spans every bin.

**The learned gate is recovered where the forward did not return it.** A run collected before the
forward returned ``event_gate_learned`` still carries $g^{\mathrm{eff}}_t$ and $o_t$. Where
$o_t = 1$ the two gates are equal, so $g_t$ is exact there; where $o_t = 0$ it is unknown and the
anchor is forced shut whatever it was. Every state count is therefore exact on such a run too.

A model built with neither a gate nor a UA mask has nothing to report, and the analysis records a
skip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.collect import CONTRACTION_AGE_COLUMN
from teb_vae.lag_attn_cfs.eval.frames import grouped_frame_entry, per_recording_labels
from teb_vae.lag_attn_cfs.gate_states import (
    FORCED_SHUT,
    GATE_OPEN_FROM,
    GATE_SHUT_BELOW,
    GATE_STATE_LABELS,
    GATE_STATE_NAMES,
    MODEL_SHUT,
    gate_states,
    run_lengths,
)

#: This analysis's own subdirectory inside the results directory.
ANALYSIS_DIRNAME = "event_gate"

#: What it writes.
STATES_FILENAME = "event_gate_states.csv"
PER_RECORDING_FILENAME = "event_gate_per_recording.csv"
BY_TIME_FILENAME = "event_gate_by_time_to_delivery.csv"
BY_CONTRACTION_FILENAME = "event_gate_by_contraction_age.csv"
FIGURE_STEM = "event_gate"

#: The per-anchor columns the collection pass writes for the gate.
APPLIED_COLUMN = "event_gate"
LEARNED_COLUMN = "event_gate_learned"
OPEN_COLUMN = "source_open"
KL_COLUMN = "kld_per_t"

#: The forecast gain read per state, the first one the table carries. The Monte Carlo gap is the
#: run's headline gap, so it is preferred over the single-draw one.
GAIN_COLUMNS: Tuple[str, ...] = ("mc_pred_gap", "pred_gap")

#: Width of a time-before-delivery bin, in hours, and the last bin edge.
TIME_BIN_HOURS = 0.5
MAX_HOURS = 12.0

#: Width of a contraction-age bin, in seconds, and the last bin edge. Five minutes covers one
#: contraction cycle in active labour.
CONTRACTION_BIN_S = 15.0
MAX_CONTRACTION_AGE_S = 300.0

#: How many quantile bins of the learned gate the KL profile uses.
GATE_BINS = 10

#: One colour per state, in state-code order, so a state has one colour on every panel.
STATE_COLORS: Tuple[str, ...] = (
    figures.COLOR_GRAY, figures.COLOR_VERMILLION, figures.COLOR_ORANGE, figures.COLOR_BLUE,
)

#: The per-recording columns resolved by cohort.
GROUPED_METRICS: Tuple[str, ...] = (
    "learned_gate_mean_open_keys",
    "applied_gate_mean",
    *(f"{name}_frac" for name in GATE_STATE_NAMES),
    "gate_kl_spearman",
)


def anchor_frame(per_anchor: pd.DataFrame) -> Optional[pd.DataFrame]:
    r"""Return the per-anchor rows with the learned gate, $o_t$ and the state attached.

    Args:
        per_anchor: The collection pass's per-anchor table.

    Returns:
        A copy carrying ``learned``, ``applied``, ``open`` and ``state`` columns, or ``None`` when
        the table has neither a gate column nor a UA-mask column, which means the model builds
        neither.
    """
    if per_anchor is None or per_anchor.empty:
        return None
    has_gate = APPLIED_COLUMN in per_anchor.columns
    has_mask = OPEN_COLUMN in per_anchor.columns
    if not (has_gate or has_mask):
        return None
    frame = per_anchor.copy()
    opened = (
        frame[OPEN_COLUMN].to_numpy(dtype=float) >= 0.5 if has_mask
        else np.ones(len(frame), dtype=bool)
    )
    if LEARNED_COLUMN in frame.columns:
        learned = frame[LEARNED_COLUMN].to_numpy(dtype=float)
    elif has_gate:
        # Equal to the applied gate wherever a key is open; unknown, and irrelevant, elsewhere.
        learned = np.where(opened, frame[APPLIED_COLUMN].to_numpy(dtype=float), np.nan)
    else:
        # A UA mask with no learned gate: the head applies $o_t$ itself.
        learned = np.ones(len(frame), dtype=float)
    applied = (
        frame[APPLIED_COLUMN].to_numpy(dtype=float) if has_gate else opened.astype(float)
    )
    states = gate_states(learned, opened if has_mask else None)
    # Where $g_t$ is unknown the anchor can only be one the mask closed.
    states[(states < 0) & ~opened] = FORCED_SHUT
    frame["learned"], frame["applied"], frame["open"] = learned, applied, opened
    frame["state"] = states
    if "epoch" in frame.columns and "anchor" in frame.columns:
        seconds = (
            frame["epoch"].to_numpy(dtype=float)
            + frame["anchor"].to_numpy(dtype=float) * SECONDS_PER_STEP
        )
        frame["hours_before_delivery"] = np.clip(-seconds / 3600.0, 0.0, None)
    return frame


def gain_column(frame: pd.DataFrame) -> Optional[str]:
    """The first of :data:`GAIN_COLUMNS` the frame carries, or ``None``."""
    return next((name for name in GAIN_COLUMNS if name in frame.columns), None)


def blink_seconds(frame: pd.DataFrame) -> np.ndarray:
    """Return the duration of every run of consecutive model-shut anchors, in seconds.

    A run is broken by a missing anchor as well as by a different state, so two shut spans on
    either side of an unscored anchor count as two runs.

    Args:
        frame: From :func:`anchor_frame`, with ``guid``, ``epoch`` and ``anchor``.

    Returns:
        One duration per run, over every segment.
    """
    if not {"guid", "epoch", "anchor"} <= set(frame.columns):
        return np.zeros(0)
    durations: List[np.ndarray] = []
    for _, segment in frame.groupby(["guid", "epoch"], sort=False):
        anchors = segment["anchor"].to_numpy(dtype=np.int64)
        if anchors.size == 0:
            continue
        dense = np.zeros(int(anchors.max() - anchors.min() + 1), dtype=bool)
        shut = segment["state"].to_numpy() == MODEL_SHUT
        dense[anchors[shut] - anchors.min()] = True
        durations.append(run_lengths(dense) * SECONDS_PER_STEP)
    return np.concatenate(durations) if durations else np.zeros(0)


def per_recording_table(frame: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Reduce the anchors to one row per recording.

    Args:
        frame: From :func:`anchor_frame`.
        labelled: The recording labels, indexed by ``guid``.

    Returns:
        One row per recording with the state shares, the gate means, the per-state KL and gain,
        the Spearman correlation of $g_t$ with $K_t$ over the anchors whose keys are open, and the
        cohort labels.
    """
    gain = gain_column(frame)
    rows: List[Dict[str, Any]] = []
    for guid, recording in frame.groupby("guid", sort=True):
        states = recording["state"].to_numpy()
        counted = states >= 0
        total = max(int(counted.sum()), 1)
        available = recording[recording["open"] & np.isfinite(recording["learned"])]
        row: Dict[str, Any] = {
            "guid": guid,
            "n_anchors": int(counted.sum()),
            "learned_gate_mean_open_keys": (
                float(available["learned"].mean()) if len(available) else float("nan")
            ),
            "applied_gate_mean": float(np.nanmean(recording["applied"].to_numpy(dtype=float))),
        }
        for code, name in enumerate(GATE_STATE_NAMES):
            in_state = recording[states == code]
            row[f"{name}_frac"] = float(len(in_state)) / float(total)
            row[f"{name}_kl_nats"] = (
                float(in_state[KL_COLUMN].mean())
                if KL_COLUMN in in_state.columns and len(in_state) else float("nan")
            )
            row[f"{name}_gain_nats"] = (
                float(in_state[gain].mean()) if gain is not None and len(in_state) else float("nan")
            )
        row["gate_kl_spearman"] = (
            float(available["learned"].corr(available[KL_COLUMN], method="spearman"))
            if KL_COLUMN in available.columns and len(available) >= 3
            and available["learned"].nunique() > 1 else float("nan")
        )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("guid") if rows else pd.DataFrame()
    for name in labelled.columns:
        table[name] = labelled[name].reindex(table.index)
    return table


def states_table(
    per_guid: pd.DataFrame, frame: pd.DataFrame, *, resamples: int, seed: int
) -> List[Dict[str, Any]]:
    """Summarise each state over recordings: its share, its KL and its forecast gain.

    Args:
        per_guid: From :func:`per_recording_table`.
        frame: From :func:`anchor_frame`, for the anchor counts.
        resamples: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        One row per state, each mean taken over recordings with a percentile interval.
    """
    rows: List[Dict[str, Any]] = []
    for code, name in enumerate(GATE_STATE_NAMES):
        row: Dict[str, Any] = {
            "state": name,
            "label": GATE_STATE_LABELS[code],
            "n_anchors": int(np.sum(frame["state"].to_numpy() == code)),
        }
        for quantity in ("frac", "kl_nats", "gain_nats"):
            column = f"{name}_{quantity}"
            values = per_guid[column].to_numpy(dtype=float) if column in per_guid else np.zeros(0)
            finite = values[np.isfinite(values)]
            interval = shared_stats.bootstrap_ci(finite, resamples=resamples, seed=seed)
            # The point is the plain mean over recordings. The bootstrap returns NaN below its
            # minimum group size, and a state's share is still a measurement on few recordings.
            row[quantity] = float(finite.mean()) if finite.size else float("nan")
            row[f"{quantity}_lo"] = interval["lo"]
            row[f"{quantity}_hi"] = interval["hi"]
            row[f"{quantity}_n_recordings"] = int(interval["n"])
        rows.append(row)
    return rows


def binned_profile(
    frame: pd.DataFrame, column: str, edges: np.ndarray
) -> pd.DataFrame:
    """Anchor-level gate means and state shares in bins of one clock.

    Args:
        frame: From :func:`anchor_frame`.
        column: The clock column to bin on.
        edges: Bin edges, ascending.

    Returns:
        One row per non-empty bin: its centre, the anchor count, the mean learned gate over the
        anchors whose keys are open, the mean applied gate and the share of each state.
    """
    if column not in frame.columns:
        return pd.DataFrame()
    clock = frame[column].to_numpy(dtype=float)
    position = np.digitize(clock, edges) - 1
    rows: List[Dict[str, Any]] = []
    for bin_index in range(len(edges) - 1):
        inside = frame[(position == bin_index) & np.isfinite(clock)]
        if not len(inside):
            continue
        available = inside[inside["open"] & np.isfinite(inside["learned"])]
        row: Dict[str, Any] = {
            "bin_lo": float(edges[bin_index]),
            "bin_hi": float(edges[bin_index + 1]),
            "bin_center": 0.5 * float(edges[bin_index] + edges[bin_index + 1]),
            "n_anchors": int(len(inside)),
            "n_recordings": int(inside["guid"].nunique()) if "guid" in inside else 0,
            "learned_gate_mean_open_keys": (
                float(available["learned"].mean()) if len(available) else float("nan")
            ),
            "applied_gate_mean": float(inside["applied"].mean()),
        }
        states = inside["state"].to_numpy()
        for code, name in enumerate(GATE_STATE_NAMES):
            row[f"{name}_frac"] = float(np.mean(states == code))
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_figure(
    frame: pd.DataFrame,
    by_time: pd.DataFrame,
    by_contraction: pd.DataFrame,
    states: Sequence[Dict[str, Any]],
    class_of: pd.Series,
    path: Path,
) -> Path:
    """Draw the six-panel gate summary and return the path written.

    Args:
        frame: From :func:`anchor_frame`.
        by_time: From :func:`binned_profile` on hours before delivery.
        by_contraction: From :func:`binned_profile` on contraction age.
        states: From :func:`states_table`.
        class_of: The clinical class of each anchor, aligned with ``frame``.
        path: The figure stem.

    Returns:
        The file written.
    """
    fig, axes = figures.panel_grid(6, n_cols=2, panel_height=2.4)
    available = frame["open"].to_numpy() & np.isfinite(frame["learned"].to_numpy(dtype=float))
    thresholds = (GATE_SHUT_BELOW, GATE_OPEN_FROM)

    # (a) The learned gate where the UA keys are open, per clinical class.
    ax = axes[0]
    classes = sorted({str(value) for value in class_of[available].dropna()})
    colours = figures.group_colors(classes) if classes else {}
    bins = np.linspace(0.0, 1.0, 41)
    if classes:
        for name in classes:
            values = frame.loc[available & (class_of.astype(str) == name).to_numpy(), "learned"]
            if len(values):
                ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.1,
                        color=colours.get(name), label=f"{name} (n={len(values)})")
    else:
        ax.hist(frame.loc[available, "learned"], bins=bins, density=True, histtype="step",
                linewidth=1.1, color=figures.COLOR_BLUE)
    for level in thresholds:
        ax.axvline(level, color=figures.COLOR_GRAY, linewidth=0.7, linestyle=":")
    ax.set_title("$g_t$ with UA keys open")
    ax.set_xlabel("$g_t$")
    ax.set_ylabel("density")
    if classes:
        ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)

    # (b) The state shares against time before delivery.
    ax = axes[1]
    if not by_time.empty:
        shares = [by_time[f"{name}_frac"].to_numpy() for name in GATE_STATE_NAMES]
        ax.stackplot(
            by_time["bin_center"], *shares, colors=STATE_COLORS, labels=GATE_STATE_LABELS,
            alpha=0.8, linewidth=0,
        )
        ax.invert_xaxis()
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=5.5, frameon=False, loc="lower left")
    ax.set_title("Gate state vs time to delivery")
    ax.set_xlabel("hours before delivery")
    ax.set_ylabel("share of anchors")
    figures.style_axes(ax)

    # (c) The two gate means against time before delivery.
    ax = axes[2]
    if not by_time.empty:
        ax.plot(by_time["bin_center"], by_time["learned_gate_mean_open_keys"],
                color=figures.COLOR_BLUE, linewidth=1.3, marker="o", markersize=2.5,
                label="$g_t$ learned, keys open")
        ax.plot(by_time["bin_center"], by_time["applied_gate_mean"],
                color=figures.COLOR_VERMILLION, linewidth=1.1, linestyle="--", marker="o",
                markersize=2.5, label="$g_t o_t$ applied")
        ax.invert_xaxis()
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=6, frameon=False)
    ax.set_title("Gate mean vs time to delivery")
    ax.set_xlabel("hours before delivery")
    ax.set_ylabel("gate")
    figures.style_axes(ax)

    # (d) The learned gate against time since the last contraction onset.
    ax = axes[3]
    if not by_contraction.empty:
        ax.plot(by_contraction["bin_center"], by_contraction["learned_gate_mean_open_keys"],
                color=figures.COLOR_BLUE, linewidth=1.3, marker="o", markersize=2.5)
        ax.set_ylim(0.0, 1.0)
    else:
        ax.text(0.5, 0.5, "no contraction-age column", transform=ax.transAxes,
                ha="center", va="center", color=figures.COLOR_GRAY)
    ax.set_title("$g_t$ vs contraction age")
    ax.set_xlabel("s since contraction onset")
    ax.set_ylabel("$g_t$, keys open")
    figures.style_axes(ax)

    # (e) The forecast gain and the KL per state, means over recordings with intervals.
    ax = axes[4]
    positions = np.arange(len(states))
    for offset, quantity, label, hatch in (
        (-0.2, "gain_nats", "forecast gain", None), (0.2, "kl_nats", "$K_t$", "//"),
    ):
        point = np.array([row[quantity] for row in states], dtype=float)
        lo = np.array([row[f"{quantity}_lo"] for row in states], dtype=float)
        hi = np.array([row[f"{quantity}_hi"] for row in states], dtype=float)
        error = np.vstack([np.clip(point - lo, 0, None), np.clip(hi - point, 0, None)])
        ax.bar(positions + offset, np.nan_to_num(point), width=0.38, color=STATE_COLORS,
               hatch=hatch, alpha=0.85, label=label, edgecolor="white", linewidth=0.5)
        ax.errorbar(positions + offset, np.nan_to_num(point), yerr=np.nan_to_num(error),
                    fmt="none", ecolor=figures.COLOR_BLACK, elinewidth=0.7, capsize=2)
    ax.axhline(0.0, color=figures.COLOR_BLACK, linewidth=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([name.split("_")[0] for name in GATE_STATE_NAMES])
    ax.set_title("Gain and $K_t$ per gate state")
    ax.set_ylabel("nats per anchor")
    ax.legend(fontsize=6, frameon=False)
    figures.style_axes(ax)

    # (f) $K_t$ against the learned gate, in quantile bins over the open anchors.
    ax = axes[5]
    open_rows = frame[available]
    if KL_COLUMN in open_rows.columns and open_rows["learned"].nunique() > 1:
        quantiles = np.unique(np.quantile(open_rows["learned"], np.linspace(0, 1, GATE_BINS + 1)))
        if quantiles.size >= 2:
            bin_of = np.clip(np.digitize(open_rows["learned"], quantiles[1:-1]), 0, quantiles.size - 2)
            grouped = open_rows.assign(_bin=bin_of).groupby("_bin")
            ax.plot(grouped["learned"].mean(), grouped[KL_COLUMN].mean(),
                    color=figures.COLOR_PURPLE, linewidth=1.3, marker="o", markersize=2.5,
                    label="$K_t$")
            gain = gain_column(open_rows)
            if gain is not None:
                twin = ax.twinx()
                twin.plot(grouped["learned"].mean(), grouped[gain].mean(),
                          color=figures.COLOR_GREEN, linewidth=1.1, linestyle="--", marker="o",
                          markersize=2.5, label="forecast gain")
                twin.set_ylabel("forecast gain (nats)", color=figures.COLOR_GREEN)
    ax.set_title("$K_t$ and gain vs $g_t$")
    ax.set_xlabel("$g_t$ decile mean, keys open")
    ax.set_ylabel("$K_t$ (nats)", color=figures.COLOR_PURPLE)
    figures.style_axes(ax)
    return figures.render_figure(fig, path)


def run_event_gate_analysis(
    context: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Report the event gate's states, its two clocks, and what each state buys.

    Args:
        context: The analysis context, read for the per-anchor and per-sample tables.
        eval_config: The validated block, for the bootstrap settings.
        output_dir: The results directory.
        probe: The loader probe's record. Unused.

    Returns:
        The protocol's keys plus the per-state table, the blink durations, the headline block and
        the files written. A model with neither a gate nor a UA mask records a skip.
    """
    collection = context.collection
    per_anchor = getattr(collection, "per_anchor", None)
    per_sample = getattr(collection, "per_sample", None)
    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    frame = anchor_frame(per_anchor)
    if frame is None:
        return {
            "n_samples": None, "composition": {},
            "plan": {"capped": False},
            "skipped": True,
            "reason": (
                "the per-anchor table carries neither event_gate nor source_open, so the model "
                "builds no posterior event gate and applies no UA mask"
            ),
            "files": [],
        }

    labelled = (
        per_recording_labels(per_sample) if per_sample is not None else pd.DataFrame()
    )
    class_of = (
        frame["guid"].map(labelled[labels.CLASS_COLUMN])
        if labels.CLASS_COLUMN in labelled.columns else pd.Series([None] * len(frame))
    ).reset_index(drop=True)
    frame = frame.reset_index(drop=True)

    resamples = int(eval_config.get("bootstrap_resamples", 2000))
    seed = int(eval_config.get("seed", 0))
    per_guid = per_recording_table(frame, labelled)
    per_guid.to_csv(directory / PER_RECORDING_FILENAME)
    states = states_table(per_guid, frame, resamples=resamples, seed=seed)
    pd.DataFrame(states).to_csv(directory / STATES_FILENAME, index=False)

    by_time = binned_profile(
        frame, "hours_before_delivery", np.arange(0.0, MAX_HOURS + TIME_BIN_HOURS, TIME_BIN_HOURS)
    )
    by_time.to_csv(directory / BY_TIME_FILENAME, index=False)
    by_contraction = binned_profile(
        frame, CONTRACTION_AGE_COLUMN,
        np.arange(0.0, MAX_CONTRACTION_AGE_S + CONTRACTION_BIN_S, CONTRACTION_BIN_S),
    )
    by_contraction.to_csv(directory / BY_CONTRACTION_FILENAME, index=False)

    blinks = blink_seconds(frame)
    written = _draw_figure(
        frame, by_time, by_contraction, states, class_of, directory / FIGURE_STEM
    )

    def recording_mean(column: str) -> float:
        """The mean over recordings of one per-recording column, ``NaN`` when none is finite."""
        if column not in per_guid:
            return float("nan")
        values = per_guid[column].to_numpy(dtype=float)
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    headline = {
        "learned_gate_mean_open_keys": recording_mean("learned_gate_mean_open_keys"),
        "applied_gate_mean": recording_mean("applied_gate_mean"),
        **{f"{name}_frac": recording_mean(f"{name}_frac") for name in GATE_STATE_NAMES},
        "model_shut_median_blink_s": float(np.median(blinks)) if blinks.size else float("nan"),
        "gate_kl_spearman": recording_mean("gate_kl_spearman"),
    }
    return {
        "n_samples": (
            int(frame.groupby(["guid", "epoch"]).ngroups)
            if {"guid", "epoch"} <= set(frame.columns) else None
        ),
        "composition": {"n_recordings": int(len(per_guid))},
        "plan": {
            "capped": False, "bootstrap_resamples": resamples, "seed": seed,
            "learned_gate_recovered": LEARNED_COLUMN not in per_anchor.columns,
            "gain_column": gain_column(frame),
        },
        "thresholds": {"gate_shut_below": GATE_SHUT_BELOW, "gate_open_from": GATE_OPEN_FROM},
        "states": states,
        "blinks": {
            "n": int(blinks.size),
            "median_s": headline["model_shut_median_blink_s"],
            "q90_s": float(np.quantile(blinks, 0.9)) if blinks.size else float("nan"),
        },
        "headline": headline,
        "grouped_frames": [
            grouped_frame_entry(ANALYSIS_DIRNAME, PER_RECORDING_FILENAME, GROUPED_METRICS)
        ],
        "files": [
            STATES_FILENAME, PER_RECORDING_FILENAME, BY_TIME_FILENAME, BY_CONTRACTION_FILENAME,
            Path(str(written)).name,
        ],
    }


__all__ = [
    "ANALYSIS_DIRNAME",
    "anchor_frame",
    "binned_profile",
    "blink_seconds",
    "per_recording_table",
    "run_event_gate_analysis",
    "states_table",
]
