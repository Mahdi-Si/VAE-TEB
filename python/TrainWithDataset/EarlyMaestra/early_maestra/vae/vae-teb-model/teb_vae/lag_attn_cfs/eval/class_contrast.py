r"""Compare the clinical classes on their attributions, over many recordings rather than one example.

The example pages show one anchor per class. A single anchor cannot say whether a pattern belongs to
the class or to that recording. This module reads the **cohort pass** of
:func:`~teb_vae.lag_attn_cfs.eval.attribution_pass.run_pass` -- a larger, class-balanced draw with
one segment per recording -- and compares the classes with one rule throughout:

1. Average each recording's anchors, so that **one recording is one unit**.
2. Per class, draw the mean over recordings with a 95% percentile bootstrap band over recordings.
3. Per class pair, draw the difference of the two class means with its own bootstrap band. The
   pairs read more severe minus less severe: HIE $-$ acidosis, HIE $-$ healthy, acidosis $-$
   healthy (:func:`~teb_vae.lag_attn.eval.labels.ordered_groups`).
4. Test each scalar per recording across classes with Kruskal--Wallis, Holm-corrected within one
   family (one method and one readout), then every class pair with Mann--Whitney $U$ and Cliff's
   $\delta$ (positive: the more severe class runs higher).

**What is compared.** Two methods, on the same recordings and anchors:

* **Integrated gradients** under ``source_null`` for :data:`COHORT_READOUTS`: the signed source
  attribution by lag, its total, its unsigned total, the centroid of its unsigned lag profile, the
  sum in each configured lag band, and the class-mean lag-by-channel maps.
* **Grad-CAM** (:mod:`~teb_vae.lag_attn_cfs.eval.gradcam`) for every main readout in three views:
  the target encoder, the source K/V stream and the lag attention. Each map is normalised to sum
  to one, so its class mean shows *where* the readout looked; its centroid and unnormalised total
  are tested.

**Read the tests as descriptive.** The model was trained on healthy recordings only, so ACIDOSIS
and HIE are out of distribution, and a class difference in attribution is a difference in how the
fitted model responds, not a physiological finding.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from teb_vae.lag_attn.nets.lag_report import SECONDS_PER_STEP
from teb_vae.lag_attn_cfs.eval import attributions as core
from teb_vae.lag_attn_cfs.eval import figures_seam as figures
from teb_vae.lag_attn_cfs.eval import gradcam
from teb_vae.lag_attn_cfs.eval._reuse import labels, stats as shared_stats
from teb_vae.lag_attn_cfs.eval.lag_axis import COEFFICIENT_LAG_AXIS_LABEL

#: The cap on the cohort draw, under ``eval_config.caps``, and its default: segments, one per
#: recording, class-balanced. 150 is 50 recordings per class, about 10 minutes on one laptop GPU
#: for the shipped transformer (see ``ATTRIBUTION.md``, section 13).
COHORT_CAP_NAME = "attribution_cohort_segments"
DEFAULT_COHORT_SEGMENTS = 150

#: The readouts the cohort integrates. Two, because an integrated-gradient call costs $n$ forwards
#: and the cohort is several times the size of the detailed draw: the divergence (how much the
#: source moved the belief) and the forecast gap (whether that helped).
COHORT_READOUTS: Tuple[str, ...] = (core.READOUT_KLD, core.READOUT_PRED_GAP)

#: Grad-CAM costs one forward and one backward, so every main readout gets it.
GRADCAM_READOUTS: Tuple[str, ...] = core.MAIN_READOUTS

BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE = 0.95
ALPHA = 0.05

COHORT_ROWS_FILENAME = "attribution_cohort_rows.csv"
COHORT_VECTORS_FILENAME = "attribution_cohort_vectors.npz"
GRADCAM_ROWS_FILENAME = "gradcam_rows.csv"
GRADCAM_VECTORS_FILENAME = "gradcam_vectors.npz"
RECORDINGS_FILENAME = "attribution_class_recordings.csv"
STATS_FILENAME = "attribution_class_stats.csv"
PAIRWISE_FILENAME = "attribution_class_pairwise.csv"
LAG_CHANNEL_FILENAME = "attribution_class_lag_channel.npz"
CLASS_FIGURE = "attribution_classes"
CLASS_MAP_FIGURE = "attribution_class_maps"
GRADCAM_FIGURE = "gradcam_classes"

PAIR_COLOURS: Tuple[str, ...] = (figures.COLOR_VERMILLION, figures.COLOR_PURPLE, figures.COLOR_BLUE)


# =============================================================================
# Per-recording reduction and bootstrap bands
# =============================================================================
def per_recording(frame: pd.DataFrame, matrix: np.ndarray) -> Tuple[List[str], List[str], np.ndarray]:
    """Average the rows of each recording.

    Args:
        frame: One row per matrix row, with ``guid`` and the class column.
        matrix: $(N, W)$ row-aligned values.

    Returns:
        ``(guids, classes, curves)`` with ``curves`` of shape $(R, W)$, one row per recording.
    """
    guids = frame["guid"].astype(str).to_numpy()
    unique = list(dict.fromkeys(guids))
    classes = [str(frame[labels.CLASS_COLUMN].to_numpy()[guids == guid][0]) for guid in unique]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        curves = np.stack([np.nanmean(np.asarray(matrix, dtype=np.float64)[guids == guid], axis=0) for guid in unique]) if unique else np.zeros((0, np.asarray(matrix).shape[1]))
    return unique, classes, curves


def by_class(classes: Sequence[str], curves: np.ndarray) -> Dict[str, np.ndarray]:
    """Split per-recording curves by class, worst class first."""
    order = labels.ordered_groups(sorted(set(classes)), labels.CLASS_COLUMN)
    names = np.asarray(classes)
    return {name: curves[names == name] for name in order}


def _resampled_means(curves: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """$B$ bootstrap means over recordings, $(B, W)$."""
    n = curves.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.stack([np.nanmean(curves[rng.integers(0, n, size=n)], axis=0) for _ in range(BOOTSTRAP_RESAMPLES)])


def _mean(curves: np.ndarray) -> np.ndarray:
    """The mean over recordings, ``NaN`` where no recording reaches an offset."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(curves, axis=0) if len(curves) else np.full(curves.shape[1], np.nan)


def _percentiles(draws: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The two-sided :data:`CONFIDENCE` percentiles of $(B, W)$ bootstrap draws, per column."""
    tail = 0.5 * (1.0 - CONFIDENCE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanquantile(draws, tail, axis=0), np.nanquantile(draws, 1.0 - tail, axis=0)


def mean_band(curves: np.ndarray, *, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mean over recordings and its percentile bootstrap band.

    Args:
        curves: $(R, W)$ per-recording curves.
        seed: The resampling seed.

    Returns:
        ``(mean, lo, hi)``, each $(W,)$; the band is ``NaN`` below
        :data:`~teb_vae.lag_attn.eval.stats.MIN_GROUP_SIZE` recordings.
    """
    mean = _mean(curves)
    if len(curves) < shared_stats.MIN_GROUP_SIZE:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    return (mean, *_percentiles(_resampled_means(curves, np.random.default_rng(seed))))


def difference_band(left: np.ndarray, right: np.ndarray, *, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The difference of two class means, left minus right, with a bootstrap band.

    Each class is resampled over its own recordings, independently.

    Returns:
        ``(difference, lo, hi)``, each $(W,)$.
    """
    difference = _mean(left) - _mean(right)
    if min(len(left), len(right)) < shared_stats.MIN_GROUP_SIZE:
        return difference, np.full_like(difference, np.nan), np.full_like(difference, np.nan)
    rng = np.random.default_rng(seed)
    return (difference, *_percentiles(_resampled_means(left, rng) - _resampled_means(right, rng)))


def class_pairs(order: Sequence[str]) -> List[Tuple[str, str]]:
    """Every class pair, more severe first, in the order ``order`` lists the classes."""
    return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]


# =============================================================================
# Per-recording metrics and their tests
# =============================================================================
def recording_metrics(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    gradcam_rows: pd.DataFrame,
    *,
    lag_seconds: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One row per recording: every tested scalar, and the families its columns fall in.

    Args:
        rows: The cohort's integrated-gradient rows.
        vectors: Their row-aligned vectors (``lag_profile``).
        gradcam_rows: The cohort's Grad-CAM rows.
        lag_seconds: The lag axis, for the lag-profile centroid.
        lag_bands: The configured lag bands; each one is a ``lagband_<band>`` column on ``rows``.

    Returns:
        ``(metrics, families)``: indexed by ``guid`` with the class column, and
        ``{family: [column, ...]}`` with one family per method and readout.
    """
    pieces: List[pd.DataFrame] = []
    families: Dict[str, List[str]] = {}
    for readout in COHORT_READOUTS:
        keep = ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy() if len(rows) else np.zeros(0, bool)
        if not keep.any():
            continue
        subset = rows[keep].copy()
        profile, _total = gradcam.normalise(np.abs(np.asarray(vectors["lag_profile"])[keep]))
        subset["lag_centroid_s"] = gradcam.centroid(profile, lag_seconds)
        names = {"source_total": "source_total", "source_abs_total": "source_abs_total", "lag_centroid_s": "lag_centroid_s"}
        names.update({f"lagband_{band}": f"lagband_{band}" for band in lag_bands if f"lagband_{band}" in subset})
        renamed = {column: f"ig_{readout}_{name}" for column, name in names.items()}
        families[f"ig:{readout}"] = list(renamed.values())
        pieces.append(subset[["guid", labels.CLASS_COLUMN, *names]].rename(columns=renamed))
    for readout in GRADCAM_READOUTS:
        subset = gradcam_rows[gradcam_rows["readout"] == readout] if len(gradcam_rows) else gradcam_rows
        if subset.empty:
            continue
        names = [f"{view}_{stat}" for view in gradcam.VIEWS for stat in ("centroid_s", "total")]
        renamed = {name: f"gradcam_{readout}_{name}" for name in names}
        families[f"gradcam:{readout}"] = list(renamed.values())
        pieces.append(subset[["guid", labels.CLASS_COLUMN, *names]].rename(columns=renamed))
    if not pieces:
        return pd.DataFrame(columns=[labels.CLASS_COLUMN]), {}
    means = [piece.groupby("guid").agg({**{c: "mean" for c in piece.columns if c not in ("guid", labels.CLASS_COLUMN)}, labels.CLASS_COLUMN: "first"}) for piece in pieces]
    metrics = means[0]
    for part in means[1:]:
        metrics = metrics.combine_first(part)
    return metrics, families


def class_tests(metrics: pd.DataFrame, families: Mapping[str, Sequence[str]], *, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The omnibus and pairwise class tests of every per-recording metric.

    Args:
        metrics: From :func:`recording_metrics`.
        families: The Holm families.
        seed: The bootstrap seed.

    Returns:
        ``(stats, pairwise)``. ``stats`` has one row per metric: per-class ``n``, mean and
        bootstrap interval, the Kruskal--Wallis statistic, $p$ and Holm $p$ within the family.
        ``pairwise`` has one row per metric and class pair: the means, their difference with a
        bootstrap interval, Mann--Whitney $p$ with Holm over the metric's pairs, and Cliff's
        $\\delta$.
    """
    order = labels.ordered_groups(sorted(metrics[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN) if len(metrics) else []
    stat_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for family, columns in families.items():
        family_rows: List[Dict[str, Any]] = []
        for column in columns:
            samples = {}
            for name in order:
                values = metrics.loc[metrics[labels.CLASS_COLUMN] == name, column].to_numpy(dtype=np.float64)
                samples[name] = values[np.isfinite(values)]
            record: Dict[str, Any] = {"family": family, "metric": column}
            for name, values in samples.items():
                interval = shared_stats.bootstrap_ci(values, confidence=CONFIDENCE, resamples=BOOTSTRAP_RESAMPLES, seed=seed)
                record.update({f"n_{name}": int(values.size), f"mean_{name}": interval["point"],
                               f"ci_lo_{name}": interval["lo"], f"ci_hi_{name}": interval["hi"]})
            testable = {name: values for name, values in samples.items() if values.size >= shared_stats.MIN_GROUP_SIZE}
            omnibus = shared_stats.kruskal_across_groups(testable)
            record.update({"n_classes": len(testable), "statistic": omnibus["statistic"], "p_value": omnibus["p_value"]})
            family_rows.append(record)
            pairs = shared_stats.pairwise_comparisons(testable) if len(testable) >= 2 else []
            adjusted = shared_stats.holm_adjust([pair["p_value"] for pair in pairs])
            for pair, p_holm in zip(pairs, adjusted):
                left, right = samples[pair["left"]], samples[pair["right"]]
                draws = np.random.default_rng(seed)
                diffs = [left[draws.integers(0, left.size, left.size)].mean() - right[draws.integers(0, right.size, right.size)].mean()
                         for _ in range(BOOTSTRAP_RESAMPLES)]
                tail = 0.5 * (1.0 - CONFIDENCE)
                pair_rows.append({
                    "family": family, "metric": column, "left": pair["left"], "right": pair["right"],
                    "n_left": pair["n_left"], "n_right": pair["n_right"],
                    "mean_left": float(left.mean()), "mean_right": float(right.mean()),
                    "mean_diff": float(left.mean() - right.mean()),
                    "diff_ci_lo": float(np.quantile(diffs, tail)), "diff_ci_hi": float(np.quantile(diffs, 1.0 - tail)),
                    "p_value": pair["p_value"], "p_holm_pairs": p_holm,
                    "cliffs_delta": pair["cliffs_delta"], "magnitude": pair["magnitude"],
                })
        for record, p_holm in zip(family_rows, shared_stats.holm_adjust([r["p_value"] for r in family_rows])):
            record["p_holm"] = p_holm
            record["significant"] = bool(np.isfinite(p_holm) and p_holm < ALPHA)
        stat_rows.extend(family_rows)
    stats_frame = pd.DataFrame(stat_rows)
    if len(stats_frame) and len(pair_rows):
        omnibus = stats_frame.set_index("metric")["p_holm"]
        for row in pair_rows:
            row["omnibus_p_holm"] = float(omnibus.get(row["metric"], np.nan))
    return stats_frame, pd.DataFrame(pair_rows)


# =============================================================================
# Figures
# =============================================================================
def _class_lines(ax: Any, x: np.ndarray, groups: Mapping[str, np.ndarray], *, seed: int, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    """Class means with bootstrap bands on one axis."""
    colours = figures.group_colors(list(groups))
    drawn = False
    for name, curves in groups.items():
        if not len(curves):
            continue
        mean, lo, hi = mean_band(curves, seed=seed)
        ax.plot(x, mean, color=colours[name], linewidth=figures.LINE_EMPHASIS, label=f"{name} (n={len(curves)})")
        ax.fill_between(x, lo, hi, color=colours[name], alpha=0.2, linewidth=0.0)
        drawn = True
    _finish(ax, drawn, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)


def _difference_lines(ax: Any, x: np.ndarray, groups: Mapping[str, np.ndarray], *, seed: int, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    """Every class pair's difference of means with its bootstrap band, and the zero line."""
    drawn = False
    for index, (left, right) in enumerate(class_pairs(list(groups))):
        if not (len(groups[left]) and len(groups[right])):
            continue
        difference, lo, hi = difference_band(groups[left], groups[right], seed=seed)
        colour = PAIR_COLOURS[index % len(PAIR_COLOURS)]
        ax.plot(x, difference, color=colour, linewidth=figures.LINE_EMPHASIS, label=f"{left} − {right}")
        ax.fill_between(x, lo, hi, color=colour, alpha=0.2, linewidth=0.0)
        drawn = True
    ax.axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN)
    _finish(ax, drawn, title=title, xlabel=xlabel, ylabel=ylabel, legend=legend)


def _finish(ax: Any, drawn: bool, *, title: str, xlabel: str, ylabel: str, legend: bool) -> None:
    """Title, labels and legend, or the empty note."""
    if not drawn:
        core._empty(ax, title)
        return
    figures.style_axes(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="best", fontsize=figures.FONT_TINY)


def _test_label(stats: pd.DataFrame, metric: str) -> str:
    """``Kruskal-Wallis p_Holm = ...`` for one metric, or empty."""
    if stats.empty or metric not in set(stats["metric"]):
        return ""
    value = float(stats.set_index("metric").loc[metric, "p_holm"])
    return f"KW p_Holm = {value:.2g}" if np.isfinite(value) else "KW: not testable"


def build_class_figure(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    metrics: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    lag_seconds: np.ndarray,
    seed: int,
    caveat: str,
) -> Any:
    r"""The integrated-gradient class comparison, one row per cohort readout.

    Left: the class-mean signed source attribution by lag, with 95% bootstrap bands over
    recordings. Middle: every class pair's difference of means with its band; a band that excludes
    zero marks lags where the two classes differ. Right: the per-recording source total by class,
    titled with the Holm-adjusted Kruskal--Wallis $p$.

    Returns:
        The figure.
    """
    figure, axes = figures.new_figure(max(len(COHORT_READOUTS), 1), 3, height_per_row=2.8, width=15.0)
    for index, readout in enumerate(COHORT_READOUTS):
        short = core.READOUT_SHORT.get(readout, readout)
        keep = ((rows["readout"] == readout) & (rows["baseline"] == core.BASELINE_SOURCE_NULL)).to_numpy() if len(rows) else np.zeros(0, bool)
        unit = str(rows.loc[keep, "unit"].iloc[0]) if keep.any() else ""
        if keep.any():
            _guids, classes, curves = per_recording(rows[keep], np.asarray(vectors["lag_profile"])[keep])
            groups = by_class(classes, curves)
        else:
            groups = {}
        _class_lines(axes[index, 0], lag_seconds, groups, seed=seed, title=f"{short}: source attribution by lag, class means",
                     xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel=f"attribution ({unit})", legend=index == 0)
        _difference_lines(axes[index, 1], lag_seconds, groups, seed=seed, title=f"{short}: class differences",
                          xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel=f"difference ({unit})", legend=index == 0)
        metric = f"ig_{readout}_source_total"
        if metric in metrics.columns:
            order = labels.ordered_groups(sorted(metrics[labels.CLASS_COLUMN].dropna().astype(str).unique()), labels.CLASS_COLUMN)
            samples = {name: metrics.loc[metrics[labels.CLASS_COLUMN] == name, metric].to_numpy(dtype=np.float64) for name in order}
            figures.violin_panel(axes[index, 2], samples, title=f"{short}: source total per recording; {_test_label(stats, metric)}",
                                 ylabel=f"attribution ({unit})", colors=figures.group_colors(order))
            axes[index, 2].axhline(0.0, color=figures.COLOR_GRAY, linewidth=figures.LINE_THIN)
        else:
            core._empty(axes[index, 2], f"{short}: source total per recording")
    figures.caveat_note(figure, caveat)
    return figure


def build_class_map_figure(lag_channel: Mapping[Tuple[str, str, str, str], Mapping[str, Any]], *, lag_seconds: np.ndarray, caveat: str) -> Any:
    r"""Which source channel at which lag, per class, and how the classes differ.

    One row per cohort readout. The class columns show the mean $|a|$ of the source attribution
    under ``source_null`` by lag and declared channel, on one shared log scale per row. The pair
    columns show the difference of those means, left minus right, on one shared linear symmetric
    scale per row, clipped at the 99th percentile of $|\Delta|$.

    Returns:
        The figure.
    """
    order = labels.ordered_groups(sorted({key[3] for key in lag_channel}), labels.CLASS_COLUMN)
    pairs = class_pairs(order)
    n_cols = max(len(order) + len(pairs), 1)
    figure, axes = figures.new_figure(max(len(COHORT_READOUTS), 1), n_cols, height_per_row=2.8, width=3.8 * n_cols)
    half = 0.5 * float(SECONDS_PER_STEP)
    for index, readout in enumerate(COHORT_READOUTS):
        short = core.READOUT_SHORT.get(readout, readout)
        fields = {name: np.asarray(lag_channel[(readout, core.BASELINE_SOURCE_NULL, core.STREAM_SOURCE, name)]["mean_abs"], dtype=np.float64).T
                  for name in order if (readout, core.BASELINE_SOURCE_NULL, core.STREAM_SOURCE, name) in lag_channel}
        differences = {(left, right): fields[left] - fields[right] for left, right in pairs if left in fields and right in fields}
        if not fields:
            for ax in axes[index]:
                core._empty(ax, short)
            continue
        extent = (float(lag_seconds[0]) - half, float(lag_seconds[-1]) + half, next(iter(fields.values())).shape[0] - 0.5, -0.5)
        shared = core.unsigned_log_norm(np.stack(list(fields.values())))
        # Linear, not symmetric-log: a log scale floored decades below the peak paints every small
        # difference at full saturation. Clipped at the 99th percentile so one cell cannot set it.
        stacked = np.abs(np.stack(list(differences.values()))) if differences else np.zeros(0)
        limit = float(np.nanpercentile(stacked, 99)) if np.isfinite(stacked).any() else 0.0
        limits = (-limit, limit) if limit > 0.0 else None
        for col, name in enumerate(order):
            if name not in fields:
                core._empty(axes[index, col], f"{short}: {name}")
                continue
            figures.heatmap_with_colorbar(
                figure, axes[index, col], fields[name], symmetric=False, norm=shared, extent=extent,
                title=f"{short}: {name}, mean |a|", xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel="source channel",
            )
        for offset, pair in enumerate(pairs):
            ax = axes[index, len(order) + offset]
            if pair not in differences:
                core._empty(ax, f"{short}: {pair[0]} − {pair[1]}")
                continue
            figures.heatmap_with_colorbar(
                figure, ax, differences[pair], symmetric=True, vlimits=limits, extent=extent,
                title=f"{short}: {pair[0]} − {pair[1]}, Δ mean |a|", xlabel=COEFFICIENT_LAG_AXIS_LABEL, ylabel="source channel",
            )
    figures.caveat_note(figure, caveat)
    return figure


def build_gradcam_figure(
    gradcam_rows: pd.DataFrame,
    gradcam_vectors: Mapping[str, np.ndarray],
    *,
    lag_seconds: np.ndarray,
    seed: int,
    caveat: str,
) -> Any:
    r"""Grad-CAM by class: one row per readout, a class-mean and a class-difference panel per view.

    Each Grad-CAM map is a share that sums to one over its offsets, so a class mean shows where the
    readout looked and a difference band that excludes zero marks offsets where two classes looked
    differently. The target view is on a minutes axis over the whole history; the source and
    attention views share the lag axis of the other lag figures.

    Returns:
        The figure.
    """
    readouts = [r for r in GRADCAM_READOUTS if len(gradcam_rows) and (gradcam_rows["readout"] == r).any()] or list(GRADCAM_READOUTS)
    figure, axes = figures.new_figure(len(readouts), 2 * len(gradcam.VIEWS), height_per_row=2.6, width=20.0)
    for index, readout in enumerate(readouts):
        short = core.READOUT_SHORT.get(readout, readout)
        keep = (gradcam_rows["readout"] == readout).to_numpy() if len(gradcam_rows) else np.zeros(0, bool)
        for col, view in enumerate(gradcam.VIEWS):
            if keep.any():
                matrix = np.asarray(gradcam_vectors[view])[keep]
                _guids, classes, curves = per_recording(gradcam_rows[keep], matrix)
                groups = by_class(classes, curves)
                x = gradcam.offset_seconds(matrix.shape[1]) / 60.0 if view == "target" else np.asarray(lag_seconds)[:matrix.shape[1]]
            else:
                groups, x = {}, np.zeros(0)
            xlabel = "offset before the anchor (min)" if view == "target" else COEFFICIENT_LAG_AXIS_LABEL
            name = gradcam.VIEW_TITLES[view]
            _class_lines(axes[index, 2 * col], x, groups, seed=seed, title=f"{name}: class means", xlabel=xlabel,
                         ylabel=f"{short}\nGrad-CAM share", legend=index == 0 and col == 0)
            _difference_lines(axes[index, 2 * col + 1], x, groups, seed=seed, title=f"{name}: class differences", xlabel=xlabel,
                              ylabel="difference of shares", legend=index == 0 and col == 0)
    figures.caveat_note(figure, caveat)
    return figure


# =============================================================================
# The block
# =============================================================================
def run_class_contrast(
    rows: pd.DataFrame,
    vectors: Mapping[str, np.ndarray],
    gradcam_rows: pd.DataFrame,
    gradcam_vectors: Mapping[str, np.ndarray],
    lag_channel: Mapping[Tuple[str, str, str, str], Mapping[str, Any]],
    *,
    directory: Path,
    lag_seconds: np.ndarray,
    lag_bands: Mapping[str, Tuple[int, int]],
    seed: int,
    caveat: str,
) -> Dict[str, Any]:
    """Write the class-comparison tables and figures of the cohort pass.

    Args:
        rows: The cohort's integrated-gradient rows.
        vectors: Their row-aligned vectors.
        gradcam_rows: The cohort's Grad-CAM rows (empty on a cell without Grad-CAM).
        gradcam_vectors: Their row-aligned profiles, keyed by view.
        lag_channel: Per-class lag-by-channel means, keyed ``(readout, baseline, stream, class)``.
        directory: The attribution directory.
        lag_seconds: The lag axis.
        lag_bands: The configured lag bands.
        seed: The bootstrap seed.
        caveat: The one-line figure note.

    Returns:
        ``n_recordings_by_class``, the significant metrics, and the files written.
    """
    metrics, families = recording_metrics(rows, vectors, gradcam_rows, lag_seconds=lag_seconds, lag_bands=lag_bands)
    stats, pairwise = class_tests(metrics, families, seed=seed)
    metrics.to_csv(directory / RECORDINGS_FILENAME)
    stats.to_csv(directory / STATS_FILENAME, index=False)
    pairwise.to_csv(directory / PAIRWISE_FILENAME, index=False)
    files = [RECORDINGS_FILENAME, STATS_FILENAME, PAIRWISE_FILENAME]
    if lag_channel:
        arrays = {"lag_seconds": np.asarray(lag_seconds, dtype=np.float64)}
        for (readout, baseline, stream, name), entry in lag_channel.items():
            arrays[f"{readout}__{baseline}__{stream}__{name}__mean"] = np.asarray(entry["mean"], dtype=np.float32)
            arrays[f"{readout}__{baseline}__{stream}__{name}__mean_abs"] = np.asarray(entry["mean_abs"], dtype=np.float32)
        np.savez_compressed(directory / LAG_CHANNEL_FILENAME, **arrays)
        files.append(LAG_CHANNEL_FILENAME)
    paths = [
        figures.render_figure(build_class_figure(rows, vectors, metrics, stats, lag_seconds=lag_seconds, seed=seed, caveat=caveat), directory / CLASS_FIGURE),
        figures.render_figure(build_class_map_figure(lag_channel, lag_seconds=lag_seconds, caveat=caveat), directory / CLASS_MAP_FIGURE),
    ]
    if len(gradcam_rows):
        paths.append(figures.render_figure(
            build_gradcam_figure(gradcam_rows, gradcam_vectors, lag_seconds=lag_seconds, seed=seed, caveat=caveat), directory / GRADCAM_FIGURE
        ))
    files.extend(Path(path).name for path in paths)
    counts = metrics[labels.CLASS_COLUMN].value_counts() if len(metrics) else pd.Series(dtype=int)
    significant = stats.loc[stats["significant"], "metric"].tolist() if len(stats) else []
    return {
        "n_recordings_by_class": {str(name): int(count) for name, count in counts.items()},
        "n_metrics_tested": int(len(stats)),
        "significant_metrics": significant,
        "test": "Kruskal-Wallis across classes, Holm within each family (method x readout); "
                "Mann-Whitney U and Cliff's delta per class pair, Holm over a metric's pairs",
        "unit": "one recording (its anchors averaged); bands are 95% percentile bootstraps over recordings",
        "files": files,
    }


__all__ = [
    "COHORT_CAP_NAME", "COHORT_READOUTS", "DEFAULT_COHORT_SEGMENTS", "GRADCAM_READOUTS",
    "build_class_figure", "build_class_map_figure", "build_gradcam_figure", "by_class",
    "class_pairs", "class_tests", "difference_band", "mean_band", "per_recording",
    "recording_metrics", "run_class_contrast",
]
