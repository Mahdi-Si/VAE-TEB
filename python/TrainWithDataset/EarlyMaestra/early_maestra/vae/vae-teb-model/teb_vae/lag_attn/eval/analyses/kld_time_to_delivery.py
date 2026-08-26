r"""How does the source-conditioned KL evolve as delivery approaches, and by class?

Every other latent analysis reduces the split to a single distribution of the per-segment KL
$\overline{K}$ -- the mean of $K_t$ over the latent dimensions and the supervised support, masked
exactly as the loss masks (warm-up prefix and, under ``kld_support='anchor'``, the final $H_d$
anchors). That distribution says *how much* source information the posterior carries, but not
*when* in gestation it carries it. This analysis resolves the same quantity against **time to
delivery** and asks whether the trajectory differs between the clinical classes.

**The time axis comes from the ``epoch`` field**, which the dataset stores as the segment start
time in **seconds relative to delivery, negative before delivery**
(``hdf5_dataset/dataset_explained_research.md``). Each 20-minute segment is placed into a
$30$-minute-wide *time-before-delivery* bin, and the per-segment $\overline{K}$ is summarised
within each (bin, group) cell.

**Two readings are drawn and one is tested.** Trajectories are drawn for both the clinical class
axis (``healthy`` / ``acidosis`` / ``hie``) and the eight canonical subgroups, because a reader
wants both cuts. The significance test runs on the **class** axis only, as requested, and is
non-parametric throughout, exactly like ``cross_subgroup``:

- **Per time window** -- a Kruskal-Wallis test across the classes present in each bin. This is what
  makes it a statement about the *trajectory*: it localises the windows in which the classes'
  $\overline{K}$ differ, rather than collapsing gestation into one number.
- **Holm across the windows** -- the per-bin omnibus tests form one family, and a correction across
  them controls the family-wise error rate. Pairwise Mann-Whitney with Cliff's delta then runs for
  the windows that survive Holm only.
- **A pooled context test** -- one Kruskal-Wallis across the classes ignoring time, reported beside
  the per-window family and explicitly flagged ``confounded_by_time``: the classes do not cover the
  time-to-delivery axis equally, so a pooled difference can be a coverage artifact rather than a
  difference in coupling. It is context, never the headline.

The statistics are the shared Layer-0 helpers in :mod:`teb_vae.lag_attn.eval.stats`, the same ones
``cross_subgroup`` uses, so a $p$-value here means exactly what it does there.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import figures, labels, masks, metrics, report, stats
from teb_vae.lag_attn.eval.collectors import CollectionPlan, collect_metrics
from teb_vae.lag_attn.eval.runner import EvalRunner, get_field

#: Subdirectory of the run directory receiving this analysis's artifacts.
ANALYSIS_DIRNAME = "kld_time_to_delivery"

#: Width of a time-before-delivery bin, in hours. Not configurable, for the same reason
#: ``cross_subgroup``'s ``alpha`` is not: it is a property of how the trajectory is resolved, and
#: an operator who could widen it could merge two windows until a difference disappeared or
#: appeared. Thirty minutes over the roughly $12.4$ h dataset gives around twenty-five windows.
BIN_WIDTH_HOURS = 0.5

#: Family-wise error rate the Holm correction across the time windows controls.
DEFAULT_ALPHA = 0.05

#: ``eval_config.caps`` key naming this analysis's retention cap. The retention is a handful of
#: scalars per sample, so it is rarely worth setting.
CAP_NAME = "kld_time_to_delivery"

#: The per-segment KL column, identical in definition to ``latent``'s ``kld_mean``: the mean of
#: $K_t$ over the latent dimensions and the KL support.
KLD_COLUMN = "kld_mean"

#: Seconds per hour, for the ``epoch``-to-hours conversion.
_SECONDS_PER_HOUR = 3600.0


# ---------------------------------------------------------------------------
# Per-batch collection
# ---------------------------------------------------------------------------
def _per_batch_kld(runner: EvalRunner, batch: Any) -> Dict[str, Any]:
    r"""Compute the per-segment KL and carry the segment's ``epoch``.

    ``kld_mean`` is built through the same Layer-0 seam ``latent`` uses -- ``masks.kld_mask`` then
    ``metrics.kld_per_dim`` then ``metrics.kld_aggregates`` -- so it is bit-for-bit the quantity
    the rest of the pipeline reports, not a re-derivation that could drift from it.

    Args:
        runner: The loaded runner.
        batch: A batch already on the compute device.

    Returns:
        Column name to a per-sample value of length $B$. ``epoch`` is present only when the batch
        carries it; a split without ``epoch`` therefore yields a frame with no ``epoch`` column,
        which the analysis reports as a clean skip rather than a failure.
    """
    model = runner.model
    outputs = runner.forward(batch)
    weight = get_field(batch, "weight")
    batch_size, seq_len = int(outputs["mu_post"].shape[0]), int(outputs["mu_post"].shape[1])
    mask_bt = masks.kld_mask(model, weight, batch_size, seq_len, device=outputs["mu_post"].device)

    aggregates = metrics.kld_aggregates(metrics.kld_per_dim(outputs, model), mask_bt)

    columns: Dict[str, Any] = {
        KLD_COLUMN: aggregates["kld_mean"],
        "n_support_steps": mask_bt.sum(dim=1),
    }
    epoch = get_field(batch, "epoch")
    if epoch is not None:
        columns["epoch"] = epoch
    return columns


# ---------------------------------------------------------------------------
# Binning and trajectory (pure)
# ---------------------------------------------------------------------------
def bin_samples(
    frame: pd.DataFrame, *, width: float = BIN_WIDTH_HOURS,
    max_hours: Optional[float] = None,
) -> pd.DataFrame:
    r"""Add the time-before-delivery bin columns, dropping rows with no usable coordinate.

    ``epoch`` is seconds relative to delivery and negative before it, so
    $\mathrm{hours\_before\_delivery} = -\mathrm{epoch} / 3600$ is non-negative for a segment
    recorded before delivery. A sample is dropped -- and counted in ``n_dropped_nonfinite`` by the
    caller -- when either its $\overline{K}$ or its ``epoch`` is non-finite, because neither a
    coordinate nor a value can be placed on the trajectory.

    Args:
        frame: The collected per-sample frame, carrying :data:`KLD_COLUMN` and ``epoch``.
        width: Bin width in hours.
        max_hours: The run's horizon -- how far before delivery a segment may be recorded and
            still be evaluated. ``None`` evaluates every segment. Applied *before* the finite
            check so a segment beyond the horizon is out of scope rather than counted in
            ``n_dropped_nonfinite``, which means something else entirely.

    Returns:
        A copy holding only the usable rows, with ``time_to_delivery_h``, ``bin`` and
        ``bin_center_h`` added. Empty when nothing is usable.
    """
    if max_hours is not None and not frame.empty and "epoch" in frame.columns:
        # ``~(hours > horizon)`` rather than ``hours <= horizon``: every comparison against NaN
        # is False, so the negation keeps the non-finite rows for the check below to drop and
        # count as what they are.
        beyond = -frame["epoch"].to_numpy(dtype=np.float64) / _SECONDS_PER_HOUR
        frame = frame[~(beyond > float(max_hours))]

    usable = frame[
        np.isfinite(frame[KLD_COLUMN].to_numpy(dtype=np.float64))
        & np.isfinite(frame["epoch"].to_numpy(dtype=np.float64))
    ].copy()
    if usable.empty:
        usable["time_to_delivery_h"] = np.zeros(0)
        usable["bin"] = np.zeros(0, dtype=np.int64)
        usable["bin_center_h"] = np.zeros(0)
        return usable

    hours_before = -usable["epoch"].to_numpy(dtype=np.float64) / _SECONDS_PER_HOUR
    # A segment at or after delivery (epoch >= 0) lands at hours_before <= 0; floor then clip keeps
    # it in the first bin rather than producing a negative index.
    n_bins = int(np.floor(float(np.nanmax(hours_before)) / width)) + 1
    bin_index = np.clip(np.floor(hours_before / width).astype(np.int64), 0, n_bins - 1)

    usable["time_to_delivery_h"] = hours_before
    usable["bin"] = bin_index
    usable["bin_center_h"] = (bin_index + 0.5) * width
    return usable


def build_trajectory(
    frame: pd.DataFrame, axis: str, *, width: float = BIN_WIDTH_HOURS
) -> pd.DataFrame:
    r"""Summarise $\overline{K}$ within each (group, bin) cell of one grouping axis.

    Args:
        frame: A frame already carrying the bin columns from :func:`bin_samples`.
        axis: The grouping column -- ``labels.CLASS_COLUMN`` or ``labels.SUBGROUP_COLUMN``.
        width: Bin width in hours, for the ``bin_center_h`` of an empty result.

    Returns:
        A long-form table with one row per (group, bin): ``group``, ``bin``, ``bin_center_h``,
        ``n`` (finite count only), ``mean``, ``q25``, ``median``, ``q75``. Rows whose group is
        absent (a segment with no class or a non-canonical subgroup) are excluded, since a segment
        with no cohort belongs to no trajectory. Quartiles rather than a standard deviation because
        these distributions are skewed, matching the grouped-variant convention.
    """
    columns = ["group", "bin", "bin_center_h", "n", "mean", "q25", "median", "q75"]
    rows: List[Dict[str, Any]] = []
    if axis not in frame.columns or frame.empty:
        return pd.DataFrame(rows, columns=columns)

    labelled = frame[frame[axis].notna()]
    for (group, bin_index), cell in labelled.groupby([axis, "bin"], sort=True):
        values = cell[KLD_COLUMN].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append({
            "group": str(group),
            "bin": int(bin_index),
            "bin_center_h": (int(bin_index) + 0.5) * width,
            "n": int(values.size),
            "mean": float(values.mean()),
            "q25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
        })
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Significance across classes, per time window (pure)
# ---------------------------------------------------------------------------
def analyse_class_trajectories(
    frame: pd.DataFrame, *, alpha: float = DEFAULT_ALPHA, width: float = BIN_WIDTH_HOURS
) -> Dict[str, Any]:
    r"""Test whether the class $\overline{K}$ trajectories differ, window by window.

    Args:
        frame: A frame already carrying the bin columns and ``labels.CLASS_COLUMN``.
        alpha: Family-wise error rate the Holm correction controls.
        width: Bin width in hours, for reporting each window's centre.

    Returns:
        The complete record: whether the test could run, the per-bin omnibus results with their
        Holm-adjusted $p$-values, the pairwise comparisons for the windows that survived, and the
        pooled (time-ignoring) context test. When fewer than two clinical classes are present the
        record carries ``tested=False`` and a reason -- the ordinary outcome on a single-class
        split, and not a failure.
    """
    classes = labels.distinct_groups(list(frame[labels.CLASS_COLUMN]))
    method = (
        "Per time-before-delivery window: Kruskal-Wallis across clinical classes, Holm step-down "
        "correction across the windows, pairwise two-sided Mann-Whitney U with Cliff's delta for "
        "the windows significant after Holm only. Non-parametric throughout. Classes with fewer "
        f"than {stats.MIN_GROUP_SIZE} finite values in a window are excluded from it and recorded. "
        "The pooled test ignores time and is confounded by unequal class coverage of the axis."
    )
    if len(classes) < 2:
        return {
            "group_column": labels.CLASS_COLUMN,
            "tested": False,
            "reason": f"fewer than two clinical classes present ({classes or 'none'})",
            "alpha": float(alpha),
            "bin_width_hours": float(width),
            "per_bin": [],
            "pairwise": {},
            "pooled": _pooled_class_test(frame),
            "method": method,
        }

    # The cohort half of the job, which is this analysis's own: which classes are in a window, in
    # which order, and which were too small to enter the test at all. The arithmetic half -- the
    # omnibus, the correction across the windows and the pairwise sweep on the survivors -- is
    # `stats.windowed_group_comparisons`, shared with every other trajectory analysis in the
    # family so that "significant" has one definition rather than one per pipeline.
    samples_by_bin: Dict[int, Dict[str, np.ndarray]] = {}
    meta_by_bin: Dict[int, Dict[str, Any]] = {}
    for bin_index in sorted(int(value) for value in frame["bin"].unique()):
        cell = frame[frame["bin"] == bin_index]
        usable, excluded = _class_samples(cell)
        samples_by_bin[int(bin_index)] = usable
        meta_by_bin[int(bin_index)] = {
            "bin_center_h": (int(bin_index) + 0.5) * float(width),
            "groups_excluded_as_too_small": excluded,
            "min_group_size": stats.MIN_GROUP_SIZE,
        }

    # ``window_field="bin"`` keeps this analysis's own published column name: its significance CSV
    # and the figures beside it have been keyed on ``bin`` since before the procedure was shared.
    outcome = stats.windowed_group_comparisons(
        samples_by_bin, meta_by_window=meta_by_bin, window_field="bin", alpha=alpha
    )
    per_bin = outcome["per_window"]
    significant = [record for record in per_bin if record["significant"]]
    return {
        "group_column": labels.CLASS_COLUMN,
        "tested": True,
        "alpha": float(alpha),
        "bin_width_hours": float(width),
        "classes": classes,
        "n_bins_tested": outcome["n_windows_tested"],
        "n_significant_bins": len(significant),
        "significant_bin_centers_h": [record["bin_center_h"] for record in significant],
        "per_bin": per_bin,
        "pairwise": outcome["pairwise"],
        "pooled": _pooled_class_test(frame),
        "method": method,
    }


def _class_samples(frame: pd.DataFrame):
    """Split one window's rows into per-class finite samples, dropping classes that are too small.

    Args:
        frame: The rows of a single time bin.

    Returns:
        ``(usable, excluded)`` -- classes with at least :data:`stats.MIN_GROUP_SIZE` finite values,
        and the sizes of those without, both in the canonical **worst-first** order. That order is
        the orientation as well as the presentation: the pairwise sweep names each pair in the
        order it receives the classes, so this is what makes every comparison read HIE vs acidosis,
        HIE vs healthy, acidosis vs healthy. The exclusion is returned rather than dropped because
        "this class had two segments in this window" is the explanation for a window the test
        skipped.
    """
    usable: Dict[str, np.ndarray] = {}
    excluded: Dict[str, int] = {}
    for group in _ordered_groups(
        labels.distinct_groups(list(frame[labels.CLASS_COLUMN])), labels.CLASS_COLUMN
    ):
        values = np.asarray(
            frame.loc[frame[labels.CLASS_COLUMN].astype(str) == group, KLD_COLUMN],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size < stats.MIN_GROUP_SIZE:
            excluded[group] = int(finite.size)
            continue
        usable[group] = finite
    return usable, excluded


def _pooled_class_test(frame: pd.DataFrame) -> Dict[str, Any]:
    """Run one time-ignoring Kruskal-Wallis across classes, flagged as confounded.

    Args:
        frame: The full binned frame.

    Returns:
        The omnibus record, tagged ``confounded_by_time`` with the reason, so it can never be read
        as the trajectory result.
    """
    usable, _ = _class_samples(frame)
    record = stats.kruskal_across_groups(usable)
    record["confounded_by_time"] = True
    record["note"] = (
        "pooled across all windows, so a difference here can be an artifact of the classes "
        "covering the time-to-delivery axis unequally rather than a difference in coupling"
    )
    return record


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _ordered_groups(groups: List[str], axis: str) -> List[str]:
    """Return the groups in the evaluation's one cohort order: worst first.

    Bound to :func:`~teb_vae.lag_attn.eval.labels.ordered_groups` rather than reimplemented -- the
    order decides which cohort is the ``left`` of every pairwise comparison as well as which
    violin is drawn first, and two copies of it would be two answers to the same question.

    Args:
        groups: The group labels present.
        axis: The grouping column, choosing the canonical order.

    Returns:
        Classes in ``hie`` / ``acidosis`` / ``healthy`` order, subgroups in reversed canonical
        order, with anything unrecognised appended alphabetically so nothing is silently dropped.
    """
    return labels.ordered_groups(groups, axis)


def _draw_trajectory_panel(ax: Any, trajectory: pd.DataFrame, axis: str, title: str) -> int:
    r"""Draw one median-with-IQR trajectory per group on a shared time axis.

    Args:
        ax: Target axes.
        trajectory: The long-form trajectory table from :func:`build_trajectory`.
        axis: The grouping column, for the colour and order conventions.
        title: Panel title.

    Returns:
        The number of groups drawn. Zero draws the empty note instead.
    """
    groups = _ordered_groups(sorted(set(trajectory["group"])), axis) if len(trajectory) else []
    if not groups:
        ax.text(
            0.5, 0.5, figures.EMPTY_NOTE, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color=figures.COLOR_GRAY,
        )
        ax.set_title(title)
        figures.style_axes(ax)
        return 0

    colours = figures.group_colors(groups)
    for group in groups:
        cell = trajectory[trajectory["group"] == group].sort_values("bin_center_h")
        if cell.empty:
            continue
        x = cell["bin_center_h"].to_numpy(dtype=np.float64)
        colour = colours.get(group, figures.COLOR_BLUE)
        ax.fill_between(
            x, cell["q25"].to_numpy(dtype=np.float64), cell["q75"].to_numpy(dtype=np.float64),
            color=colour, alpha=0.15, linewidth=0,
        )
        ax.plot(
            x, cell["median"].to_numpy(dtype=np.float64), marker="o", markersize=3,
            color=colour, linewidth=1.4, label=f"{group} (n={int(cell['n'].sum())})",
        )
    ax.set_title(title)
    ax.set_xlabel("Time before delivery (hours)")
    ax.set_ylabel("$\\overline{K}$ per segment (nats)")
    # Delivery (0 h) sits at the right, so the eye reads left-to-right toward delivery.
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="best")
    figures.style_axes(ax)
    return len(groups)


def _write_trajectory_figure(
    by_class: pd.DataFrame, by_subgroup: pd.DataFrame, directory: Path
) -> str:
    """Draw the two trajectory panels -- by class and by subgroup -- and write them.

    Args:
        by_class: The class-axis trajectory table.
        by_subgroup: The subgroup-axis trajectory table.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    figure, axes = figures.new_figure(2, height_per_row=3.2)
    try:
        _draw_trajectory_panel(
            axes[0, 0], by_class, labels.CLASS_COLUMN,
            "Per-segment KL vs time before delivery, by clinical class",
        )
        _draw_trajectory_panel(
            axes[1, 0], by_subgroup, labels.SUBGROUP_COLUMN,
            "Per-segment KL vs time before delivery, by subgroup",
        )
        return str(figures.render_figure(figure, directory / "trajectory"))
    finally:
        figures.plt.close(figure)


def _write_significance_figure(record: Dict[str, Any], directory: Path) -> str:
    r"""Draw the per-window significance and its pairwise effect sizes.

    Two panels because they answer different questions. The upper is $-\log_{10}$ of the
    Holm-adjusted $p$ per time window against the $\alpha$ line -- *in which windows do the classes
    differ*. The lower is the Cliff's delta of every surviving class pair in those windows -- *by
    how much, and which way*.

    Args:
        record: The significance record from :func:`analyse_class_trajectories`.
        directory: The analysis directory.

    Returns:
        The path written.
    """
    figure, axes = figures.new_figure(2, height_per_row=3.2)
    try:
        if not record.get("tested"):
            axes[0, 0].text(
                0.5, 0.5, record.get("reason", figures.EMPTY_NOTE), transform=axes[0, 0].transAxes,
                ha="center", va="center", fontsize=9, color=figures.COLOR_GRAY,
            )
            axes[0, 0].set_title("Class trajectory significance (not tested)")
            figures.style_axes(axes[0, 0])
            axes[1, 0].axis("off")
            return str(figures.render_figure(figure, directory / "significance"))

        # The mark itself is the shared one: the same bars against the same threshold line that
        # every family in this repository is read against, so two figures of two different
        # families cannot come to draw "is there anything there" two different ways. The strip
        # filters the untestable windows out for itself, which is why the whole list goes in.
        drawn = figures.significance_strip(
            axes[0, 0],
            [row["bin_center_h"] for row in record["per_bin"]],
            [row["p_holm"] for row in record["per_bin"]],
            alpha=float(record["alpha"]),
            bin_width=float(record["bin_width_hours"]),
            title="Class difference by time window (Kruskal-Wallis, Holm)",
            xlabel="Time before delivery (hours)",
        )
        if drawn:
            # Delivery sits at the right, so the eye reads left to right toward it.
            axes[0, 0].invert_xaxis()
        else:
            # The panel already says it has no finite value to draw; what only this analysis knows
            # is *why*, and the title is where it fits without a second note beside the first.
            axes[0, 0].set_title(
                "Class difference by time window -- no window had two testable classes"
            )

        _draw_pairwise_heatmap(figure, axes[1, 0], record)
        return str(figures.render_figure(figure, directory / "significance"))
    finally:
        figures.plt.close(figure)


def _draw_pairwise_heatmap(figure: Any, ax: Any, record: Dict[str, Any]) -> None:
    """Draw the Cliff's delta of every surviving class pair across the significant windows.

    Args:
        figure: The parent figure, for the colourbar.
        ax: Target axes.
        record: The significance record.
    """
    rows: List[Dict[str, Any]] = []
    centre_by_bin = {int(row["bin"]): row["bin_center_h"] for row in record["per_bin"]}
    for bin_key, comparisons in record["pairwise"].items():
        for item in comparisons:
            rows.append({
                "bin_center_h": centre_by_bin.get(int(bin_key), float("nan")),
                "pair": f"{item['left']} vs {item['right']}",
                "cliffs_delta": item.get("cliffs_delta", float("nan")),
            })
    if not rows:
        figures.heatmap_with_colorbar(
            figure, ax, np.zeros((0, 0)),
            title="Cliff's delta (no window survived Holm)",
            symmetric=True, colorbar_label="Cliff's delta",
        )
        return

    pairs = sorted({row["pair"] for row in rows})
    centres = sorted({row["bin_center_h"] for row in rows})
    field = np.full((len(pairs), len(centres)), np.nan)
    for row in rows:
        field[pairs.index(row["pair"]), centres.index(row["bin_center_h"])] = row["cliffs_delta"]
    figures.heatmap_with_colorbar(
        figure, ax, field, title="Cliff's delta for surviving class pairs",
        symmetric=True, colorbar_label="Cliff's delta",
    )
    figures.label_rows(ax, pairs)
    ax.set_xticks(np.arange(len(centres)))
    ax.set_xticklabels([f"{value:g}" for value in centres], fontsize=6)
    ax.set_xlabel("Time before delivery (hours)")


# ---------------------------------------------------------------------------
# Emission (model-free core)
# ---------------------------------------------------------------------------
def emit_analysis(
    frame: pd.DataFrame,
    output_dir: Any,
    *,
    composition: Optional[Dict[str, int]] = None,
    plan: Optional[Dict[str, Any]] = None,
    alpha: float = DEFAULT_ALPHA,
    width: float = BIN_WIDTH_HOURS,
    max_hours: Optional[float] = None,
) -> Dict[str, Any]:
    r"""Bin, summarise, test and emit -- everything after the forward pass, and model-free.

    Split out so the whole analysis can be exercised from a hand-built per-sample frame, exactly
    as ``cross_subgroup`` is: the collection needs a model, but the binning and the statistics do
    not.

    Args:
        frame: The per-sample frame, carrying :data:`KLD_COLUMN`, ``epoch`` and the class /
            subgroup label columns.
        output_dir: The run's results directory.
        composition: The per-file draw composition, for the coverage record.
        plan: The collection plan record, for the coverage record.
        alpha: Family-wise error rate.
        width: Bin width in hours.
        max_hours: The run's ``max_hours_before_delivery`` horizon, passed through to
            :func:`bin_samples`. ``None`` evaluates every segment.

    Returns:
        The JSON-safe summary for ``summary.json``. A ``skipped`` record -- leaving no directory --
        when the frame is empty, carries no ``epoch`` column, has no usable rows, or holds no
        class and no subgroup labels at all.
    """
    n_total = int(len(frame))
    if frame.empty:
        return _skip("the collected frame was empty", n_total)
    if "epoch" not in frame.columns:
        return _skip("the batch carried no 'epoch' field, so time to delivery is unavailable", n_total)

    binned = bin_samples(frame, width=width, max_hours=max_hours)
    n_dropped = n_total - int(len(binned))
    if binned.empty:
        return _skip("no sample had a finite KL and a finite epoch", n_total)

    has_class = bool(labels.distinct_groups(list(binned[labels.CLASS_COLUMN])))
    has_subgroup = bool(labels.distinct_groups(list(binned[labels.SUBGROUP_COLUMN])))
    if not (has_class or has_subgroup):
        return _skip(
            "no clinical class or subgroup labels were present, so there is no trajectory to draw",
            n_total,
        )

    directory = Path(output_dir) / ANALYSIS_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)

    per_sample_columns = [
        "sample_index", "guid", "source_file", labels.CLASS_COLUMN, labels.SUBGROUP_COLUMN,
        KLD_COLUMN, "epoch", "time_to_delivery_h", "bin", "bin_center_h", "n_support_steps",
    ]
    present = [name for name in per_sample_columns if name in binned.columns]
    binned[present].to_csv(directory / "per_sample.csv", index=False)

    by_class = build_trajectory(binned, labels.CLASS_COLUMN, width=width)
    by_subgroup = build_trajectory(binned, labels.SUBGROUP_COLUMN, width=width)
    by_class.to_csv(directory / "trajectory_by_class.csv", index=False)
    by_subgroup.to_csv(directory / "trajectory_by_subgroup.csv", index=False)

    significance = analyse_class_trajectories(binned, alpha=alpha, width=width)
    _write_significance_tables(significance, directory)
    with open(directory / f"{ANALYSIS_DIRNAME}.json", "w", encoding="utf-8") as handle:
        json.dump(report.json_safe(significance), handle, indent=2)

    figure_paths = [
        _write_trajectory_figure(by_class, by_subgroup, directory),
        _write_significance_figure(significance, directory),
    ]

    summary: Dict[str, Any] = {
        "skipped": False,
        "n_samples": int(len(binned)),
        "n_dropped_nonfinite": int(n_dropped),
        "composition": dict(composition or {}),
        "plan": plan,
        "bin_width_hours": float(width),
        "n_bins": int(binned["bin"].max()) + 1 if len(binned) else 0,
        "trajectory": {
            "by_class": _trajectory_summary(by_class),
            "by_subgroup": _trajectory_summary(by_subgroup),
        },
        "significance": {
            "group_column": labels.CLASS_COLUMN,
            "tested": significance["tested"],
            "reason": significance.get("reason"),
            "alpha": float(alpha),
            "n_bins_tested": significance.get("n_bins_tested", 0),
            "n_significant_bins": significance.get("n_significant_bins", 0),
            "significant_bin_centers_h": significance.get("significant_bin_centers_h", []),
            "largest_effects": _largest_effects(significance),
            "pooled": significance["pooled"],
            "method": significance["method"],
        },
        "figures": figure_paths,
    }
    logger.info(
        f"kld_time_to_delivery: {summary['n_samples']} sample(s) over "
        f"{summary['n_bins']} window(s) of {width:g} h; class test "
        f"{'ran' if significance['tested'] else 'skipped'} -- "
        f"{summary['significance']['n_significant_bins']} of "
        f"{summary['significance']['n_bins_tested']} window(s) differ after Holm"
    )
    return summary


def _skip(reason: str, n_total: int) -> Dict[str, Any]:
    """Return the skip record, and log it.

    Args:
        reason: Why the analysis did not run.
        n_total: How many samples were collected before the skip.

    Returns:
        The skip summary. No directory is created, so a skipped analysis leaves nothing behind.
    """
    logger.warning(f"kld_time_to_delivery: skipped -- {reason}")
    return {"skipped": True, "reason": reason, "n_samples": int(n_total)}


def _trajectory_summary(trajectory: pd.DataFrame) -> Dict[str, Any]:
    """Return a compact record of one trajectory table, for the summary."""
    return {
        "n_groups": int(trajectory["group"].nunique()) if len(trajectory) else 0,
        "groups": sorted(set(trajectory["group"])) if len(trajectory) else [],
        "n_cells": int(len(trajectory)),
    }


def _write_significance_tables(record: Dict[str, Any], directory: Path) -> None:
    """Write the per-window omnibus table and the pairwise table.

    Args:
        record: The significance record.
        directory: The analysis directory.
    """
    per_bin_rows = [
        {
            "bin": row["bin"],
            "bin_center_h": row["bin_center_h"],
            "n_classes": row["n_groups"],
            "n_total": sum(row["n_per_group"].values()),
            "statistic": row["statistic"],
            "p_value": row["p_value"],
            "p_holm": row.get("p_holm", float("nan")),
            "significant": row.get("significant", False),
            "alpha": row.get("alpha", float("nan")),
        }
        for row in record["per_bin"]
    ]
    pd.DataFrame(
        per_bin_rows,
        columns=["bin", "bin_center_h", "n_classes", "n_total", "statistic", "p_value",
                 "p_holm", "significant", "alpha"],
    ).to_csv(directory / "significance.csv", index=False)

    centre_by_bin = {int(row["bin"]): row["bin_center_h"] for row in record["per_bin"]}
    pairwise_rows: List[Dict[str, Any]] = []
    for bin_key, comparisons in record["pairwise"].items():
        for item in comparisons:
            pairwise_rows.append({
                "bin": int(bin_key),
                "bin_center_h": centre_by_bin.get(int(bin_key), float("nan")),
                "left": item["left"],
                "right": item["right"],
                "n_left": item["n_left"],
                "n_right": item["n_right"],
                "u_statistic": item.get("u_statistic", float("nan")),
                "p_value": item["p_value"],
                "cliffs_delta": item["cliffs_delta"],
                "magnitude": item["magnitude"],
            })
    pd.DataFrame(
        pairwise_rows,
        columns=["bin", "bin_center_h", "left", "right", "n_left", "n_right", "u_statistic",
                 "p_value", "cliffs_delta", "magnitude"],
    ).to_csv(directory / "pairwise.csv", index=False)


def _largest_effects(record: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    r"""Return the surviving class pairs with the largest absolute effect size.

    Ranked by $|\delta|$ rather than by $p$: the smallest $p$ is usually the pair with the most
    samples, not the largest difference.

    Args:
        record: The significance record.
        limit: How many to report.

    Returns:
        The strongest comparisons, largest first.
    """
    centre_by_bin = {int(row["bin"]): row["bin_center_h"] for row in record["per_bin"]}
    effects: List[Dict[str, Any]] = []
    for bin_key, comparisons in record.get("pairwise", {}).items():
        for item in comparisons:
            if np.isfinite(item.get("cliffs_delta", float("nan"))):
                effects.append({
                    "bin_center_h": centre_by_bin.get(int(bin_key), float("nan")),
                    "left": item["left"],
                    "right": item["right"],
                    "cliffs_delta": float(item["cliffs_delta"]),
                    "magnitude": item["magnitude"],
                    "p_value": float(item["p_value"]),
                })
    effects.sort(key=lambda item: abs(item["cliffs_delta"]), reverse=True)
    return effects[: int(limit)]


# ---------------------------------------------------------------------------
# Registry entry point
# ---------------------------------------------------------------------------
def run_kld_time_to_delivery_analysis(
    runner: EvalRunner,
    loader: Any,
    *,
    eval_config: Dict[str, Any],
    output_dir: Any,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect per-segment KL and time to delivery, then draw and test the trajectory.

    Args:
        runner: The loaded runner.
        loader: The eval dataloader.
        eval_config: The validated ``eval_config`` block.
        output_dir: The run's results directory.
        probe: The loader probe's record, for the sample count and per-file grouping.

    Returns:
        The headline summary for ``summary.json``, or a ``skipped`` record on a split that carries
        no ``epoch`` or no cohort labels.
    """
    caps = eval_config.get("caps") or {}
    n_total = int((probe or {}).get("n_samples") or 0)
    plan = (
        CollectionPlan.build(
            n_total, caps.get(CAP_NAME), int(eval_config.get("seed", 0)),
            groups=(probe or {}).get("source_files"),
        )
        if n_total
        else None
    )

    collected = collect_metrics(
        runner, loader, _per_batch_kld,
        max_samples=eval_config.get("max_samples"), plan=plan,
        progress_label="kld_time_to_delivery",
    )
    return emit_analysis(
        collected.frame, output_dir,
        composition=collected.composition, plan=collected.plan,
        max_hours=eval_config.get("max_hours_before_delivery"),
    )
