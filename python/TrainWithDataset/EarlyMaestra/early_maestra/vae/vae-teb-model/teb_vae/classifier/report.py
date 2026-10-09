r"""Figures and ``summary.md`` of a classifier run (SPEC §11.10-§11.11; the P2 and P4 subsets).

``report`` runs after :func:`teb_vae.classifier.metrics.evaluate` and draws **every** figure from
tables only: ``evaluation/tables/*`` (``metrics``, ``roc_points``, ``thresholds``, ``alarms``,
``inclusion.csv``), ``predictions/guids.parquet`` (PR curves) and ``cohort/*`` (block C). Evaluate
renders nothing. Figures go through :mod:`teb_vae.classifier.figstyle` (the classifier's style over the VAE
figure seam): the style is configured once, each ``eval.figure_formats`` entry is looped with ``set_figure_format``,
``render_figure`` lays out, saves and closes, and every panel shows ``EMPTY_NOTE`` on empty input instead of raising.
Afterwards ``results.artifacts`` in ``evaluation/summary.json`` is rebuilt, so the verify gate sees the figures.

Pages with one panel per model wrap into a grid (:func:`_panels`); pages with several panels per model are facet grids
(:func:`_facets`: one row per model, named in a strip right of the row). Time-resolved figures (P4) share one time axis
(hours before delivery inverted, delivery on the right), carry the primary model's n strip under the first column and
one note line (what a hollow marker means; the exclusion counts are in ``inclusion.csv``).

Figure text is kept to one title, short panel titles, axis labels, one key under the panels (:class:`_Once`,
:func:`_key`) and at most one note line; counts, statistics and definitions live in the tables, ``summary.md`` and
``SPEC.md``.

Every per-fold test value (thin fold lines, fold bands, ``fold_<k>/`` pages, fold means in ``summary.md``) is on the
weighted fold population of SPEC §11.7: a test GUID that $K$ folds' test splits hold weighs $1/K$ in each
(:func:`_fold_w`); val is unweighted. An evaluation without ``results.fold_weighting`` is drawn as it is and labelled
unweighted (:func:`_fold_phrase`).

Nothing heavy is imported at module load (the seam pulls torch), so ``verify`` can import
:func:`FIGURE_REGISTRY` torch-free.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.classifier import figstyle
from teb_vae.classifier.metrics import (
    TYPES, _read_json, _sel, _wquantile, classifier_cfg, fold_weights_of, guid_timeline, infer_stride, pool_rows,
    primary_alpha, unique_cohort,
)

COHORT_OVERVIEW = "cohort/cohort_overview"
COHORT_SUBGROUPS = "cohort/cohort_subgroups"
COHORT_TIME_BINS = "cohort/cohort_time_bins"
COHORT_RANKED_LENGTHS = "cohort/cohort_ranked_lengths"
COHORT_COVERAGE = "cohort/cohort_coverage"
COHORT_GAPS = "cohort/cohort_gaps"
COHORT_CLOCKS = "cohort/cohort_clocks"
ROC_GUID = "roc/roc_guid"
PR_GUID = "roc/pr_guid"
THRESHOLD_DRIFT = "thresholds/threshold_drift"
# P4 (§11.12 blocks M, R2-R8, T3, T4, A); ``{axis}`` over ``eval.time_axes``, ``{policy}`` over :func:`_m1_policies`.
METRIC_TYPES = "metric_types/metric_types_{axis}_{policy}"
SEGMENT_INSTANTANEOUS = "metric_types/segment_instantaneous_{axis}"
ROC_CHECKPOINTS_CUMULATIVE = "roc/roc_checkpoints_cumulative"
ROC_CHECKPOINTS_OVERALL = "roc/roc_checkpoints_overall"
ROC_CHECKPOINTS_SNAPSHOT = "roc/roc_checkpoints_snapshot"
ROC_SEGMENT = "roc/roc_segment"
PR_CHECKPOINTS = "roc/pr_checkpoints"
AUROC_VS_TIME = "roc/auroc_vs_time_{axis}"
DECISION_HORIZON = "thresholds/decision_horizon"
THRESHOLD_STABILITY = "thresholds/threshold_stability"
METRIC_TYPE_COMPARISON = "thresholds/metric_type_comparison"
LEAD_TIME = "alarms/lead_time"
FALSE_ALARMS = "alarms/false_alarms"
COHORT_FIGURES = (COHORT_OVERVIEW, COHORT_SUBGROUPS, COHORT_TIME_BINS, COHORT_RANKED_LENGTHS,
                  COHORT_COVERAGE, COHORT_GAPS, COHORT_CLOCKS)
SUMMARY_MD = "summary.md"
#: The columns of the time-resolved (``point != 'n/a'``) metrics rows the figures read; ~1.7 M rows per model at scale.
TR_COLS = ["model_id", "seed", "fold", "split", "level", "subgroup", "subgroup_value", "axis", "t", "point",
           "metric_type", "denominator", "policy_id", "metric", "value", "ci_lo", "ci_hi", "n_pos", "n_neg"]
ALARM_COLS = ["model_id", "seed", "fold", "split", "policy_id", "rule", "guid", "y", "alarmed", "lead_time_h",
              "time_to_first_alarm_h"]
INCL_COLS = ["analysis", "model_id", "seed", "split", "fold", "axis", "n_included", "n_excluded", "reasons"]


def _m1_policies(ev: Dict[str, Any]) -> List[str]:
    """M1 figure policies: the primary and every policy decided at a time point (``at`` in hours)."""
    return list(dict.fromkeys([ev["primary_policy"], *(p["id"] for p in ev["thresholds"] if p["at"] != "end")]))


#: P6 blocks register here from their own sections (no shared literal to edit). ``AXIS_BUILDERS``: stem template
#: (``{axis}``; ``{policy}`` = the primary policy) -> builder called with ``axis=``. ``EXPECTED_WHEN``: stem (or
#: template) -> predicate on the config; the stem is expected (registry, verify) only when it holds. ``CORE_EXTRA``:
#: pooled stems (templates bound to to_delivery and the primary policy) that ``val/`` and ``fold_<k>/`` also render.
AXIS_BUILDERS: Dict[str, Callable[..., Any]] = {}
EXPECTED_WHEN: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
CORE_EXTRA: List[str] = []
#: ``evaluation/tables/<name>.parquet`` tables P6 blocks add; :func:`load_tables` loads each as ``T[name]``.
EXTRA_TABLES: List[str] = []


def _always(_: Any) -> bool:
    return True


def _figure_set(c: Dict[str, Any]) -> Dict[str, Callable[..., Any]]:
    """Every pooled stem for the config -> its builder (per-axis / per-policy stems bound here)."""
    ev = c["eval"]
    out = {s: f for s, f in _BUILDERS.items() if EXPECTED_WHEN.get(s, _always)(c)}
    for axis in ev.get("time_axes", []):  # absent in the minimal configs verify's tests pass
        out |= {METRIC_TYPES.format(axis=axis, policy=p): partial(_metric_types, axis=axis, pid=p) for p in _m1_policies(ev)}
        out[SEGMENT_INSTANTANEOUS.format(axis=axis)] = partial(_segment_instantaneous, axis=axis)
        out[AUROC_VS_TIME.format(axis=axis)] = partial(_auroc_vs_time, axis=axis)
        out |= {s.format(axis=axis, policy=ev["primary_policy"]): partial(f, axis=axis)
                for s, f in AXIS_BUILDERS.items() if EXPECTED_WHEN.get(s, _always)(c)}
    return out


def FIGURE_REGISTRY(cfg: Any) -> List[str]:  # noqa: N802 - the spec's name
    """Figure stems (relative to ``evaluation/figures``, no extension) expected for ``cfg``.

    The pooled test set of every P2/P4 figure; ``val/`` repeats the core set (R1, R6, R2 cumulative,
    M1 of the primary policy on to_delivery); ``fold_<k>/`` repeats R1, R2 and that M1 per fold
    when ``eval.per_fold_figures``.
    """
    c = classifier_cfg(cfg)
    ev = c["eval"]
    pooled = _figure_set(c)
    core = [ROC_CHECKPOINTS_CUMULATIVE, *([METRIC_TYPES.format(axis="to_delivery", policy=ev["primary_policy"])]
                                          if "to_delivery" in ev.get("time_axes", []) else [])]
    core += [s for s in (t.format(axis="to_delivery", policy=ev["primary_policy"]) for t in CORE_EXTRA) if s in pooled]
    stems = [*pooled, *(f"val/{s}" for s in (ROC_GUID, PR_GUID, *core))]
    if ev["per_fold_figures"]:
        stems += [f"fold_{k}/{s}" for k in c["run"]["folds"] for s in (ROC_GUID, *core)]
    return stems


def _seam() -> Any:
    """The classifier's figure style (:mod:`teb_vae.classifier.figstyle`), which falls through to the VAE seam."""
    from teb_vae.classifier import figstyle

    return figstyle


def _order(groups: Any, axis: str) -> List[str]:
    from teb_vae.lag_attn.eval.labels import ordered_groups

    return ordered_groups(list(groups), axis)


def _empty(ax: Any) -> None:
    fs = _seam()
    ax.text(0.5, 0.5, fs.EMPTY_NOTE, transform=ax.transAxes, ha="center", va="center",
            fontstyle="italic", color=fs.FAINT, fontsize=fs.FONT_LABEL)
    fs.style_axes(ax)


def _figure(n_rows: int, n_cols: int, height: float = 2.6, width: Optional[float] = None) -> Any:
    """``(fig, axes)`` of ``n_rows`` x ``n_cols`` panels under constrained layout; ``width`` defaults to 2.55 in per
    column (at least 4.4 in, at most :data:`figstyle.WINDOWS_FIGURE_WIDTH`)."""
    import matplotlib.pyplot as plt

    n_rows, n_cols = max(1, n_rows), max(1, n_cols)
    w = width or min(max(4.4, 2.55 * n_cols + 0.5), _seam().WINDOWS_FIGURE_WIDTH)
    return plt.subplots(n_rows, n_cols, squeeze=False, figsize=(w, height * n_rows + 0.5), layout="constrained")


def _panels(n: int, ncols: int = 3, size: float = 2.5) -> tuple:
    """``(fig, axes)`` of ``n`` roughly square panels wrapped into at most ``ncols`` columns; the unused cells are
    hidden, ``axes`` is the flat list of the ``n`` used ones."""
    nc = max(1, min(ncols, n))
    fig, axes = _figure(-(-max(n, 1) // nc), nc, size, width=size * nc + 0.6)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:max(n, 1)])


def _all_empty(n_rows: int = 1, n_cols: int = 1) -> Any:
    fig, axes = _figure(n_rows, n_cols)
    for ax in axes.flat:
        _empty(ax)
    return fig


def _models(*frames: pd.DataFrame) -> List[tuple]:
    f = next((x for x in frames if {"model_id", "seed"} <= set(x.columns) and len(x)), None)
    return [] if f is None else sorted(f[["model_id", "seed"]].drop_duplicates().itertuples(index=False, name=None))


def _who(m: Any, sd: Any) -> str:
    """A model's name on a figure: ``model · seed 42``, ``model · ensemble``, or the bare name of a seedless one."""
    sd = str(sd)
    return str(m) if sd in ("na", "nan", "None", "") else f"{m} · ensemble" if sd == "ens" else f"{m} · seed {sd}"


def _legend(ax: Any, loc: str = "best", **kw: Any) -> None:
    """A legend inside ``ax`` for entries that belong to that panel only (an AUC, a mean), on a white backing so it
    reads over a grid line. Keys shared by several panels go to the figure key instead (:class:`_Once`, :func:`_key`).
    """
    if kw.get("handles") or ax.get_legend_handles_labels()[0]:
        ax.legend(loc=loc, frameon=True, framealpha=0.88, facecolor="white", edgecolor="none", borderpad=0.4,
                  fontsize=_seam().FONT_SMALL, **kw)


def _key(ax: Any) -> None:
    """Move every legend entry of ``ax`` to its figure's one key (drawn under the panels)."""
    _seam().add_key(ax.figure, *ax.get_legend_handles_labels())


class _Once:
    """One key per figure: the entries of the first axes that has any go to the figure key (:func:`_key`); every
    later call is ignored, for panels (one per model) that share the same keys."""

    def __init__(self) -> None:
        self.done = False

    def __call__(self, ax: Any, **kw: Any) -> None:
        """:func:`_key` of ``ax`` unless an earlier call took one."""
        if self.done or not ax.get_legend_handles_labels()[0]:
            return
        self.done = True
        _key(ax)


def _row_label(ax: Any, text: str) -> None:
    """A facet strip right of ``ax`` naming its row (a model on a faceted page)."""
    fs = _seam()
    ax.annotate(text, xy=(1.0, 0.5), xycoords="axes fraction", xytext=(7, 0), textcoords="offset points",
                rotation=-90, ha="left", va="center", fontsize=fs.FONT_LABEL, fontweight="semibold", color=fs.INK,
                bbox={"boxstyle": "square,pad=0.45", "fc": fs.STRIP, "ec": "none"}, annotation_clip=False)


def _facets(n_rows: int, n_cols: int, *, row_h: float = 1.95, width: Optional[float] = None,
            strip: Tuple[int, ...] = (), sharex: Any = "all") -> tuple:
    """A faceted page: ``n_rows`` x ``n_cols`` data panels and, when ``strip`` names columns, a thin n strip row under
    those columns (hidden elsewhere, where the last data row keeps its tick labels). Returns ``(fig, grid, strips)``:
    ``grid`` is ``(n_rows, n_cols)``, ``strips`` the strip axes per column (None where hidden)."""
    import matplotlib.pyplot as plt

    ratios = [1.0] * max(1, n_rows) + ([0.3] if strip else [])
    w = width or min(3.7 * n_cols + 1.3, _seam().WINDOWS_FIGURE_WIDTH)
    fig, axes = plt.subplots(len(ratios), n_cols, sharex=sharex, squeeze=False, gridspec_kw={"height_ratios": ratios},
                             figsize=(w, row_h * sum(ratios) + 0.9), layout="constrained")
    grid, strips = axes[:max(1, n_rows)], [None] * n_cols
    if strip:
        for j in range(n_cols):
            if j in strip:
                strips[j] = axes[-1, j]
            else:
                axes[-1, j].set_visible(False)
                grid[-1, j].tick_params(labelbottom=True)
    return fig, grid, strips


def _class_hist(ax: Any, frame: pd.DataFrame, col: str, xlabel: str, legend: bool = True) -> None:
    """Filled step histogram of ``col`` per clinical class; empty note when nothing is finite."""
    fs, v = _seam(), frame.dropna(subset=[col]) if col in frame else frame.iloc[:0]
    if v.empty:
        return _empty(ax)
    bins = np.histogram_bin_edges(v[col].astype(float), bins=15)
    order = _order(v["clinical_class"].unique(), "clinical_class")
    colors = fs.group_colors(order)
    for g in order:
        x = v.loc[v["clinical_class"] == g, col].astype(float)
        ax.hist(x, bins=bins, histtype="stepfilled", color=colors[g], alpha=0.18, lw=0)
        ax.hist(x, bins=bins, histtype="step", color=colors[g], lw=fs.LINE_REGULAR * 1.3, label=g)
    ax.set(xlabel=xlabel, ylabel="GUIDs")
    if legend:
        _key(ax)
    fs.style_axes(ax, grid="y")


# ---- block C ----------------------------------------------------------------------------------
def _bars(ax: Any, x: Any, h: Any, color: Any, **kw: Any) -> None:
    """Bars with a white edge, so adjacent bars stay apart."""
    ax.bar(x, h, color=color, edgecolor="white", linewidth=0.6, **kw)


def _cohort_overview(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C4: segments per GUID, segments vs hours before delivery, GUIDs per subgroup, and the GUID, segment and
    excluded-segment totals (the exclusion reasons are in ``summary.md``, section 2, and ``cohort/``)."""
    fs, seg, gd = _seam(), T["seg"], T["gd"]
    if seg.empty or gd.empty:
        return _all_empty(2, 2)
    fig, ((a, b), (cc, d)) = _figure(2, 2, 2.5, width=fs.FIGURE_WIDTH)
    n = seg.groupby("guid").size()
    a.hist(n, bins=np.arange(n.min(), n.max() + 2) - 0.5, color=fs.BLUE, edgecolor="white", linewidth=0.6)
    a.axvline(n.mean(), ls="--", color=fs.INK, lw=fs.LINE_REGULAR, label=f"mean {n.mean():.1f}")
    a.axvline(n.median(), ls=":", color=fs.ORANGE, lw=fs.LINE_REGULAR * 1.4, label=f"median {n.median():.0f}")
    a.set(xlabel="segments per GUID", ylabel="GUIDs", title="Segments per GUID")
    _legend(a, loc="upper right")
    bin_h = c["eval"]["bin_h"]
    b.hist(seg["hours_to_delivery"], bins=np.arange(0, seg["hours_to_delivery"].max() + bin_h, bin_h),
           color=fs.BLUE, edgecolor="white", linewidth=0.6)
    b.invert_xaxis()
    b.set(xlabel="hours before delivery", ylabel="segments", title="Segments over time")
    counts = gd["source_file"].value_counts()
    order = _order(counts.index, "subgroup")
    colors = fs.group_colors(order)
    _bars(cc, range(len(order)), counts[order], [colors[g] for g in order])
    cc.set_xticks(range(len(order)), order, rotation=30, ha="right", fontsize=fs.FONT_SMALL)
    cc.set(ylabel="GUIDs", title="GUIDs per subgroup")
    excl = int(T["allseg"]["exclusion_reason"].replace("", np.nan).notna().sum()) if len(T["allseg"]) else 0
    d.axis("off")
    for i, (v, name) in enumerate(((len(gd), "GUIDs"), (len(seg), "segments"), (excl, "excluded segments"))):
        d.text(0.08, 0.86 - 0.3 * i, f"{v:,}", transform=d.transAxes, va="top", fontsize=16, fontweight="bold",
               color=fs.INK if i < 2 else fs.ROSE)
        d.text(0.08, 0.86 - 0.3 * i - 0.15, name, transform=d.transAxes, va="top", fontsize=fs.FONT_LABEL,
               color=fs.MUTED)
    for ax in (a, b, cc):
        fs.style_axes(ax, grid="y")
    return fig


def _cohort_subgroups(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C5: GUIDs (left panel) and segments (right panel) per subgroup."""
    fs, seg, gd = _seam(), T["seg"], T["gd"]
    if gd.empty:
        return _all_empty()
    fig, axes = _figure(1, 2, 2.8, width=fs.FIGURE_WIDTH)
    g, s = gd["source_file"].value_counts(), seg["source_file"].value_counts()
    order = _order(g.index, "subgroup")
    colors = fs.group_colors(order)
    for ax, v, what in ((axes[0, 0], g, "GUIDs"), (axes[0, 1], s, "segments")):
        y, v = np.arange(len(order)), v.reindex(order, fill_value=0)
        ax.barh(y, v, color=[colors[k] for k in order], edgecolor="white", height=0.72)
        for yi, vi in zip(y, v):
            ax.text(vi, yi, f" {vi:,}", va="center", fontsize=fs.FONT_SMALL, color=fs.MUTED)
        ax.set_yticks(y, order if ax is axes[0, 0] else [""] * len(order), fontsize=fs.FONT_SMALL)
        ax.invert_yaxis()
        ax.set(xlabel=what, title=f"{what.capitalize()} per subgroup")
        ax.margins(x=0.15)
        fs.style_axes(ax, grid="x")
    return fig


def _cohort_time_bins(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C6: segments per time bin stacked by class, then by subgroup (class x cs x bg)."""
    fs, seg, bin_h = _seam(), T["seg"], c["eval"]["bin_h"]
    if seg.empty:
        return _all_empty(2, 1)
    fig, axes = _figure(2, 1, 2.2, width=fs.FIGURE_WIDTH)
    b = (seg["hours_to_delivery"] // bin_h) * bin_h + bin_h / 2
    for ax, col, axis, title in ((axes[0, 0], "clinical_class", "clinical_class", "By class"),
                                 (axes[1, 0], "source_file", "subgroup", "By subgroup")):
        t = pd.crosstab(b, seg[col])
        order = _order(t.columns, axis)
        colors, bottom = fs.group_colors(order), np.zeros(len(t))
        for g in order:
            _bars(ax, t.index, t[g], colors[g], width=0.92 * bin_h, bottom=bottom, label=g)
            bottom += t[g].to_numpy()
        ax.invert_xaxis()
        ax.set(xlabel="hours before delivery", ylabel="segments", title=title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fs.FONT_SMALL)
        fs.style_axes(ax, grid="y")
    return fig


def _cohort_ranked_lengths(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C7: GUIDs ranked by segment count, coloured by class, with the mean."""
    from matplotlib.patches import Patch

    fs, seg, gd = _seam(), T["seg"], T["gd"]
    if seg.empty or gd.empty:
        return _all_empty()
    fig, axes = _figure(1, 1, 2.8, width=fs.FIGURE_WIDTH)
    ax = axes[0, 0]
    n = seg.groupby("guid").size().sort_values(ascending=False)
    cls = gd.set_index("guid")["clinical_class"].reindex(n.index)
    colors = fs.group_colors(cls.dropna().unique())
    ax.bar(np.arange(len(n)), n, width=0.85, color=[colors.get(k, fs.FAINT) for k in cls], lw=0)
    ax.axhline(n.mean(), ls="--", color=fs.INK, lw=fs.LINE_REGULAR)
    fs.add_key(fig, [Patch(color=colors[k]) for k in _order(colors, "clinical_class")] + [ax.lines[-1]],
               [*_order(colors, "clinical_class"), f"mean {n.mean():.1f}"])
    ax.set(xlabel="GUID (ranked by length)", ylabel="segments", xlim=(-1, len(n)))
    fs.style_axes(ax, grid="y")
    return fig


def _cohort_coverage(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C8: GUIDs (grouped by class, sorted by span) x slots on the delivery axis: present / excluded / absent."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fs, a, gd = _seam(), T["allseg"], T["gd"]
    stride = infer_stride(T["seg"], T["manifest"]) if len(T["seg"]) else None
    a = a[a["guid"].isin(gd["guid"])] if len(a) and len(gd) else a.iloc[:0]
    if a.empty or not stride:
        return _all_empty()
    col = (-a["t_end_s"] / stride).round().astype(int).clip(lower=0).to_numpy()
    span = a.groupby("guid")["t_end_s"].agg(lambda t: t.max() - t.min())
    classes = _order(gd["clinical_class"].unique(), "clinical_class")
    g = gd.set_index("guid").assign(span=span).reset_index()
    g["rank"] = g["clinical_class"].map({k: i for i, k in enumerate(classes)})
    rows = g.sort_values(["rank", "span"], ascending=[True, False])["guid"].tolist()
    n_col = col.max() + 1
    state = np.zeros((len(rows), n_col))
    row = a["guid"].map({k: i for i, k in enumerate(rows)}).to_numpy()
    state[row, n_col - 1 - col] = np.where(a["excluded"].astype(bool), 1, 2)
    fig, axes = _figure(1, 1, 3.8, width=fs.FIGURE_WIDTH)
    ax = axes[0, 0]
    absent = "#F2F3F5"
    ax.imshow(state, aspect="auto", interpolation="nearest", vmin=0, vmax=2,
              cmap=ListedColormap([absent, fs.ROSE, fs.BLUE]), extent=[n_col * stride / 3600, 0, len(rows), 0])
    edges = np.cumsum([int((g["clinical_class"] == k).sum()) for k in classes])
    for e in edges[:-1]:
        ax.axhline(e, color="white", lw=2.0)
    ax.set_yticks(edges - np.diff(np.r_[0, edges]) / 2, classes)
    ax.tick_params(axis="y", length=0)
    ax.set(xlabel="hours before delivery", ylabel="GUIDs, longest span first")
    fs.add_key(fig, [Patch(color=fs.BLUE), Patch(color=fs.ROSE), Patch(facecolor=absent, edgecolor=fs.FAINT)],
               ["present", "excluded", "absent"])
    fs.style_axes(ax, grid="none")
    return fig


def _timeline(T: Dict[str, Any]) -> pd.DataFrame:
    if T["seg"].empty or T["gd"].empty:
        return pd.DataFrame()
    return guid_timeline(T["seg"], T["gd"], infer_stride(T["seg"], T["manifest"]))


def _class_rate_bars(ax: Any, rates: pd.Series, ylabel: str, title: str) -> None:
    """One bar per class (worst first) of a fraction, with its value printed over the bar."""
    fs = _seam()
    order = _order(rates.index, "clinical_class")
    colors = fs.group_colors(order)
    _bars(ax, order, rates[order], [colors[k] for k in order], width=0.65)
    for k in order:
        ax.text(k, rates[k], f"{rates[k]:.2f}", ha="center", va="bottom", fontsize=fs.FONT_SMALL, color=fs.MUTED)
    ax.set(ylabel=ylabel, ylim=(0, 1.1), title=title)
    fs.style_axes(ax, grid="y")


def _cohort_gaps(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C9: largest within-GUID gap, contiguous runs, late coverage, per class."""
    tl = _timeline(T)
    if tl.empty:
        return _all_empty(1, 3)
    fig, axes = _figure(1, 3, 2.5)
    a, b, d = axes[0]
    _class_hist(a, tl, "max_gap_h", "largest within-GUID gap (h)")
    _class_hist(b, tl, "n_runs", "contiguous runs per GUID", legend=False)
    a.set_title("Largest gap")
    b.set_title("Contiguous runs")
    _class_rate_bars(d, tl.groupby("clinical_class")["late"].mean(), "fraction of GUIDs", "Data in the last hour")
    return fig


def _cohort_clocks(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """C10: TLO / second-stage availability, admission TLO and labour duration, per class."""
    fs, tl = _seam(), _timeline(T)
    if tl.empty:
        return _all_empty(1, 3)
    fig, axes = _figure(1, 3, 2.5)
    a, b, d = axes[0]
    rates = tl.groupby("clinical_class")[["has_tlo", "has_ss"]].mean()
    order, x = _order(rates.index, "clinical_class"), np.arange(len(rates))
    _bars(a, x - 0.19, rates.loc[order, "has_tlo"], fs.BLUE, width=0.38, label="has TLO")
    _bars(a, x + 0.19, rates.loc[order, "has_ss"], fs.ORANGE, width=0.38, label="has second stage")
    a.set_xticks(x, order)
    a.set(ylabel="fraction of GUIDs", ylim=(0, 1.25), title="Clock availability")
    _legend(a, loc="upper right")
    fs.style_axes(a, grid="y")
    _class_hist(b, tl, "admission_tlo_h", "onset to first segment (h)")
    _class_hist(d, tl, "labour_h", "labour duration (h)", legend=False)
    b.set_title("Admission after onset")
    d.set_title("Labour duration")
    return fig


# ---- blocks R and T ---------------------------------------------------------------------------
def _policy_color(c: Dict[str, Any], pid: str) -> str:
    """One colour per policy across every panel, keyed on the config's policy order."""
    fs, ids = _seam(), [p["id"] for p in c["eval"]["thresholds"]]
    return fs.LINE_PALETTE[ids.index(pid) % len(fs.LINE_PALETTE)] if pid in ids else fs.FAINT


def _op_points(ax: Any, met: pd.DataFrame, c: Dict[str, Any], m: str, sd: str, fold: str) -> List[str]:
    """● realised test (FPR, sens) per policy, on its own basis population, and ◦ the validation-chosen one. The points
    carry no legend entries (the policy colours are the same in every panel: see :func:`_policy_key`); returns the
    policy ids drawn."""
    pts = _sel(met, model_id=m, seed=sd, level="guid", fold=fold)
    pts = pts[pts["subgroup"].isna() & pts["metric"].isin(["sens", "fpr"]) & pts["policy_id"].notna()
              & (pts["policy_id"] != "oracle")]
    drawn = []
    for pid, g in pts.groupby("policy_id"):
        color = _policy_color(c, pid)
        for split, face, edge in (("val", "white", color), ("test", color, figstyle.EDGE)):
            q = g[g["split"] == split].set_index("metric")["value"]
            if {"sens", "fpr"} <= set(q.index):
                ax.plot(q["fpr"], q["sens"], "o", ms=5.5, mfc=face, mec=edge if split == "test" else color,
                        mew=0.9 if split == "val" else 0.6, ls="none", zorder=4)
                drawn += [pid] if split == "test" else []
    return drawn


def _policy_key(fig: Any, c: Dict[str, Any], policies: Any) -> None:
    """The operating-point entries of the figure key: a dot per policy (config order) and ``hollow: validation``."""
    from matplotlib.lines import Line2D

    fs, ids = _seam(), _pids(c, policies)
    if not ids:
        return
    dots = [Line2D([], [], marker="o", ms=5.5, ls="none", color=_policy_color(c, p), mec=figstyle.EDGE) for p in ids]
    dots.append(Line2D([], [], marker="o", ms=5.5, ls="none", mfc="white", mec=fs.MUTED))
    fs.add_key(fig, dots, [*ids, "hollow: validation"])


def _alpha_lines(ax: Any, thr: pd.DataFrame, m: str, sd: str) -> None:
    fs = _seam()
    for a in sorted(_sel(thr, model_id=m, seed=sd)["alpha"].dropna().unique()) if len(thr) else []:
        ax.axvline(a, ls=(0, (3, 3)), color=fs.FAINT, lw=fs.LINE_HAIRLINE, zorder=1)


def _stat(ax: Any, text: str, loc: str = "lower right") -> None:
    """A short statistic (an AUC) in a corner of ``ax``, on a white backing."""
    fs = _seam()
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    y, va = (0.04, "bottom") if "lower" in loc else (0.96, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs.FONT_LABEL, color=fs.INK, zorder=6,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "none", "alpha": 0.88})


def _stat_list(ax: Any, items: List[tuple], loc: str = "lower right") -> None:
    """One coloured value per model (``(colour, text)``), stacked in a corner of ``ax``: the per-panel AUCs of an
    overlay whose model names are in the figure key."""
    fs = _seam()
    x, ha = (0.97, "right") if "right" in loc else (0.03, "left")
    for i, (color, text) in enumerate(items[::-1] if "lower" in loc else items):
        y = 0.04 + 0.085 * i if "lower" in loc else 0.96 - 0.085 * i
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom" if "lower" in loc else "top",
                fontsize=fs.FONT_SMALL, color=color, fontweight="semibold", zorder=6,
                bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8})


def _unit_square(ax: Any, diagonal: bool = True) -> None:
    """A ROC-style unit square: equal aspect, limits [0, 1], the chance diagonal."""
    fs = _seam()
    if diagonal:
        ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE, zorder=1)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.02), xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect("equal", adjustable="box")


# ---- fold weighting (SPEC §11.7) ----
# Every fold's test split holds the same augmented healthy GUIDs (``shared_test``), so a fold's own test set is not
# the pooled population (77 % shared healthy GUIDs against 25 % pooled in the 2026-10-07 run). Every per-fold test
# value is on the weighted fold population instead: a row weighs $w = 1/K$, $K$ the number of folds whose test split
# holds its GUID ($w = 0$ for a shared GUID under ``data.shared_test_policy: exclude``); val stays unweighted. The
# metrics tables carry it (``results.fold_weighting``); the curves report draws from the prediction tables apply it
# here (:func:`_fold_w`).
def _fold_weighted(T: Dict[str, Any]) -> bool:
    """Whether the evaluation's per-fold rows are on the weighted fold population (``results.fold_weighting`` of
    ``evaluation/summary.json`` is ``inverse_k``). An evaluation made before 2026-10-08 has no flag: its per-fold rows
    hold every shared test GUID in every fold."""
    return (T["summary"].get("results") or {}).get("fold_weighting") == "inverse_k"


def _fold_w(T: Dict[str, Any], c: Dict[str, Any], frame: pd.DataFrame, split: str) -> np.ndarray:
    """Row weights of the per-fold populations of ``frame`` (every fold of one model/seed/split; split it by fold
    after): :func:`metrics.fold_weights_of` on test for a weighted evaluation, else 1, so the report's own per-fold
    curves follow the convention of the evaluation's per-fold rows they are drawn next to. Val stays unweighted, as
    ``fold_weights_of`` keeps it: each fold's thresholds were selected on its unweighted val population."""
    if split != "test" or not _fold_weighted(T) or not len(frame):
        return np.ones(len(frame))
    return fold_weights_of(frame, c["data"]["shared_test_policy"], split)


def _fold_phrase(T: Dict[str, Any], c: Dict[str, Any], split: str = "test") -> str:
    """The note phrase of a figure with per-fold test values (thin fold lines, a fold band, a ``fold k`` page): how a
    shared test GUID counts in them. Empty on val, whose per-fold rows stay unweighted (:func:`_fold_w`)."""
    if split != "test":
        return ""
    if not _fold_weighted(T):
        return "Folds unweighted: shared test GUIDs count in every fold."
    if c["data"]["shared_test_policy"] == "exclude":
        return "Folds: shared test GUIDs left out, as pooled."
    return "Folds: a shared test GUID weighs 1/K (K: folds that hold it)."


def _fold_def(T: Dict[str, Any], c: Dict[str, Any], short: bool = False) -> str:
    """The ``summary.md`` definition line of the per-fold test values (fold mean ± SD, min–max, per-fold columns): in
    full under the headline (§3), ``short`` (a pointer to it) in every other section that reports them."""
    excl = c["data"]["shared_test_policy"] == "exclude"
    if not _fold_weighted(T):
        return ("Per-fold test values: unweighted, every shared test GUID in every fold (see §3)." if short else
                "Per-fold test values (fold mean ± SD, min-max, per-fold columns) are unweighted: this evaluation "
                "predates fold weighting (no `fold_weighting` in `summary.json`), so each fold's test split holds every "
                "shared test GUID and the fold means are not comparable with the pooled values.")
    if short:
        return (f"Per-fold test values: {'shared test GUIDs left out' if excl else 'a shared test GUID weighs 1/K'} "
                "(see §3).")
    return ("Per-fold test values (fold mean ± SD, min-max, per-fold columns) are on the weighted fold population (SPEC "
            "§11.7): " + ("a shared test GUID weighs 0, as pooling drops it" if excl else
                          "a shared test GUID weighs 1/K in each of the K folds whose test split holds it")
            + ", so the fold means are comparable with the pooled values. Per-fold val values are unweighted.")


def _count_spec(a: Any) -> str:
    """The format of the counts ``a``: whole numbers, or one decimal where fold weighting made a per-fold count
    fractional."""
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return ".0f" if np.allclose(a, np.round(a)) else ".1f"


def _roc_guid(T: Dict[str, Any], c: Dict[str, Any], split: str = "test", fold: Optional[str] = None,
              level: str = "guid", variant: str = "score_final") -> Any:
    """R1 (and R5 with ``level='segment', variant='segment'``): pooled: thin per-fold curves,
    vertical average ± SD (and the per-fold min-max envelope under ``fold_band: minmax``) where
    stored, pooled curve with its cluster-bootstrap band; per fold: that fold's curve. GUID level:
    operating points ◦ val / ● test per policy; α lines dashed. One panel per model, AUC in its corner; the shared
    entries are in the figure key."""
    fs, roc, met, policies, key = _seam(), T["roc"], T["metrics"], set(), _Once()
    models = _models(_sel(roc, level=level, split=split))
    fig, axes = _panels(len(models), 3, 2.55)
    if not models:
        _empty(axes[0])
    for ax, (m, sd) in zip(axes, models):
        r = _sel(roc, model_id=m, seed=sd, split=split, level=level)
        au = _sel(met, model_id=m, seed=sd, split=split, level=level, metric="auroc")
        name = "pooled ROC"
        if fold is None:
            for i, (f, g) in enumerate(_sel(r, variant=variant).groupby("fold")):
                if f != "pooled":
                    ax.plot(g["fpr"], g["tpr"], color=fs.MUTED, lw=fs.LINE_THIN, alpha=0.35,
                            label="per fold" if i == 0 else None)
            v = _sel(r, variant=f"{variant}:vavg")
            if len(v):
                if c["eval"]["fold_band"] == "minmax":
                    ax.fill_between(v["fpr"], v["tpr_min"], v["tpr_max"], color=fs.MUTED, alpha=0.1, lw=0,
                                    label="fold min-max")
                ax.fill_between(v["fpr"], v["tpr"] - v["tpr_sd"], v["tpr"] + v["tpr_sd"], color=fs.ORANGE, alpha=0.15,
                                lw=0)
                ax.plot(v["fpr"], v["tpr"], color=fs.ORANGE, lw=fs.LINE_REGULAR, label="fold mean ± SD")
            b = _sel(r, variant=f"{variant}:band")
            if len(b):
                ax.fill_between(b["fpr"], b["tpr_lo"], b["tpr_hi"], color=fs.BLUE, alpha=0.16, lw=0, label="95% band")
            p, pa = _sel(r, fold="pooled", variant=variant), _sel(au, fold="pooled")
        else:
            p, pa, name = _sel(r, fold=fold, variant=variant), _sel(au, fold=fold), f"fold {fold} ROC"
        if len(p) and len(pa):
            ax.plot(p["fpr"], p["tpr"], color=fs.BLUE, lw=fs.LINE_EMPHASIS * 2, label=name, zorder=3)
            _stat(ax, f"AUC {pa['value'].iloc[0]:.3f}")
        ax.set_title(_who(m, sd))
        if not ax.lines:
            _empty(ax)
            continue
        key(ax)
        if level == "guid":
            policies |= set(_op_points(ax, met, c, m, sd, "pooled" if fold is None else fold))
            _alpha_lines(ax, T["thr"], m, sd)
        _unit_square(ax)
        ax.set(xlabel="FPR (1 − specificity)", ylabel="sensitivity")
        fs.style_axes(ax)
    _policy_key(fig, c, policies)
    _tag(fig, f"{'Segment' if level == 'segment' else 'GUID'} ROC", split=split, fold=fold)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _pr_guid(T: Dict[str, Any], c: Dict[str, Any], split: str = "test", **_: Any) -> Any:
    """R6: GUID final-score PR, thin per-fold (on the weighted fold population, :func:`_fold_w`) and pooled, pooled AP
    and the prevalence in the panel corner, the prevalence baseline dashed. One panel per model."""
    from sklearn.metrics import precision_recall_curve

    fs, gdp, met, key = _seam(), T["guids"], T["metrics"], _Once()
    models = _models(gdp)
    fig, axes = _panels(len(models), 3, 2.55)
    if not models:
        _empty(axes[0])
    for ax, (m, sd) in zip(axes, models):
        g = _sel(gdp, model_id=m, seed=sd, split=split)
        units = g.assign(unit=g["guid"], score=g["score_final_cal"], w=_fold_w(T, c, g, split))
        ap = _sel(met, model_id=m, seed=sd, split=split, level="guid", metric="auprc")
        for i, (_, f) in enumerate(units.groupby("fold")):
            if f["y"].nunique() == 2:
                pr, rc, _ = precision_recall_curve(f["y"], f["score"], sample_weight=f["w"])
                ax.plot(rc, pr, color=fs.MUTED, lw=fs.LINE_THIN, alpha=0.35, label="per fold" if i == 0 else None)
        p = pool_rows(units, c["data"]["shared_test_policy"], split)
        ax.set_title(_who(m, sd))
        if p["y"].nunique() == 2:
            pr, rc, _ = precision_recall_curve(p["y"], p["score"])
            pa = _sel(ap, fold="pooled")["value"]
            ax.plot(rc, pr, color=fs.BLUE, lw=fs.LINE_EMPHASIS * 2, label="pooled PR", zorder=3)
            ax.axhline(p["y"].mean(), ls=(0, (3, 3)), color=fs.MUTED, lw=fs.LINE_HAIRLINE, label="prevalence")
            _stat(ax, f"AP {pa.iloc[0] if len(pa) else float('nan'):.3f}\nprevalence {p['y'].mean():.2f}", "lower left")
        if not ax.lines:
            _empty(ax)
            continue
        key(ax)
        _unit_square(ax, diagonal=False)
        ax.set(xlabel="recall (sensitivity)", ylabel="precision (PPV)")
        fs.style_axes(ax)
    _tag(fig, "GUID precision-recall", split=split)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _threshold_drift(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """T2: val vs test sensitivity and FPR per fold and policy; test FPR - α per fold (dots) and pooled (●). One row
    per model (named in its facet strip)."""
    fs, thr, met = _seam(), T["thr"], T["metrics"]
    models = _models(thr)
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=2.35, width=8.8, sharex=False)
    seen = set()
    for row, (m, sd) in zip(grid, models):
        t = _sel(thr, model_id=m, seed=sd, level="guid")
        t = t[t["skipped"].isna()] if "skipped" in t else t
        te = _sel(met, model_id=m, seed=sd, level="guid", split="test")
        te = te[te["subgroup"].isna()] if len(te) else te
        pids = _pids(c, t["policy_id"].unique())
        for i, pid in enumerate(pids):
            color = _policy_color(c, pid)
            test = _sel(te, policy_id=pid).pivot_table(index="fold", columns="metric", values="value")
            j = _sel(t, policy_id=pid).set_index("fold")[["val_sens", "val_fpr"]].join(test, how="inner")
            if {"sens", "fpr"} <= set(j.columns):
                row[0].plot(j["val_sens"], j["sens"], "o", ms=4.5, color=color, mec=figstyle.EDGE, mew=0.5)
                row[1].plot(j["val_fpr"], j["fpr"], "o", ms=4.5, color=color, mec=figstyle.EDGE, mew=0.5)
                seen.add(pid)
            if "fpr_overshoot" in j and "pooled" in test.index:
                row[2].plot(np.full(len(j), i) + np.linspace(-0.15, 0.15, len(j)), j["fpr_overshoot"], "o", ms=3.5,
                            color=color, alpha=0.55, mew=0)
                row[2].plot(i, test.loc["pooled", "fpr_overshoot"], "D", ms=6, color=color, mec=fs.INK, mew=0.6)
        for ax, what in ((row[0], "sensitivity"), (row[1], "FPR")):
            if not ax.lines:
                _empty(ax)
                continue
            ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE)
            ax.set(xlabel=f"validation {what}", ylabel=f"test {what}", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
            ax.set_aspect("equal", adjustable="box")
            fs.style_axes(ax)
        _alpha_lines(row[1], thr, m, sd)
        if row[2].lines:
            row[2].axhline(0, ls=(0, (3, 3)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
            row[2].set_xticks(range(len(pids)), pids, rotation=30, ha="right", fontsize=fs.FONT_SMALL)
            row[2].set(ylabel="test FPR − α")
            fs.style_axes(row[2], grid="y")
        else:
            _empty(row[2])
        _row_label(row[2], _who(m, sd))
    for ax, title in zip(grid[0], ("Sensitivity: validation vs test", "FPR: validation vs test", "FPR overshoot")):
        ax.set_title(title)
    from matplotlib.lines import Line2D

    ids = _pids(c, seen)
    fs.add_key(fig, [Line2D([], [], marker="o", ls="none", ms=5, color=_policy_color(c, p), mec=figstyle.EDGE) for p in ids]
               + [Line2D([], [], marker="D", ls="none", ms=5, color=fs.MUTED, mec=fs.INK)], [*ids, "pooled"])
    _tag(fig, "Threshold drift")
    fs.caveat_note(fig, text=_fold_phrase(T, c))  # its per-fold values are test ones
    return fig


# ---- P4: time-resolved figures (blocks M, R2-R8, T3, T4, A) ---------------------------------------
#: The rate lines: sensitivity ○, specificity □, FPR △, in :data:`figstyle.RATE_COLORS`.
RATES = (("sens", "sensitivity", figstyle.RATE_COLORS["sens"], "o"), ("spec", "specificity", figstyle.RATE_COLORS["spec"], "s"),
         ("fpr", "FPR", figstyle.RATE_COLORS["fpr"], "^"))
#: The three metric types as row labels (their populations are defined in ``SPEC.md``, "Metric types").
TYPE_LABEL = {"instantaneous": "Instantaneous", "committed_cumulative": "Committed\ncumulative",
              "committed_overall": "Committed\noverall"}
AXIS_LABEL = {"to_delivery": "hours before delivery", "from_onset": "hours since labour onset",
              "rel_second_stage": "hours from second-stage onset", "position": "segment index",
              "elapsed": "hours since first segment"}
ROC_KIND = {"committed_cumulative": "Committed cumulative", "committed_overall": "Committed overall",
            "snapshot": "Snapshot"}
#: Under-power marker of the time-resolved figures: a hollow marker is a bin with fewer GUIDs of a class than needed.
UNDER_BIN = ("Hollow, coloured ring: fewer than {n} GUIDs of a class in the bin; hollow, black ring: no estimate "
             "in the bin (the line is joined over it)")


def _tr(T: Dict[str, Any], **eq: Any) -> pd.DataFrame:
    """The time-resolved rows of one (level, axis, split), memoised: the table is ~1.7 M rows per model
    and the per-axis figures (4+ M1 policies, M6, R7) all start from the same slice."""
    memo = T.setdefault("tr_memo", {})
    key = tuple(sorted(eq.items()))
    if key not in memo:
        memo[key] = _sel(T["tr"], **eq)
    return memo[key]


def _one(fold: Optional[str]) -> str:
    return "pooled" if fold is None else str(fold)


def _tag(fig: Any, *head: str, split: str = "test", fold: Optional[str] = None) -> None:
    """The figure title: ``head`` (what the figure shows, a model or a policy) and, off the pooled test set,
    ``validation`` or ``fold k``. Nothing is drawn when both are empty."""
    text = " · ".join([*filter(None, head), *(["validation"] if split == "val" else [f"fold {fold}"] if fold is not None
                                               else [])])
    if text:
        _seam().set_title(fig, text)


def _policy(c: Dict[str, Any], pid: Optional[str]) -> Dict[str, Any]:
    return next((p for p in c["eval"]["thresholds"] if p["id"] == pid), {})


def _pids(c: Dict[str, Any], present: Any) -> List[str]:
    """Config-ordered policy ids among ``present``."""
    return [p["id"] for p in c["eval"]["thresholds"] if p["id"] in set(present)]


def _bottom(grid: Any, strips: List[Any], j: int) -> Any:
    """The lowest visible axes of column ``j`` of a :func:`_facets` page (its strip, else its last data row)."""
    return strips[j] if strips and strips[j] is not None else grid[-1, j]


def _wrap(n: int, ncols: int = 2) -> tuple:
    """``(rows, cols)`` of ``n`` panels wrapped into at most ``ncols`` columns."""
    nc = max(1, min(ncols, n))
    return -(-max(n, 1) // nc), nc


def _time_lines(ax: Any, axis: str, pol: Optional[Dict[str, Any]] = None) -> None:
    """The dashed grey basis time point of ``pol`` (on its own axis) and the dotted second-stage onset."""
    fs = _seam()
    if axis == "rel_second_stage":
        ax.axvline(0.0, ls=":", color=fs.INK, lw=fs.LINE_REGULAR, label="second-stage onset", zorder=1)
    if pol and pol.get("at", "end") != "end" and pol.get("axis", "to_delivery") == axis:
        ax.axvline(float(pol["at"]), ls=(0, (4, 2)), color=fs.MUTED, lw=fs.LINE_REGULAR,
                   label=f"decision {float(pol['at']):g} h", zorder=1)


def _xlim(axes: Any, t: pd.Series, axis: str, pol: Optional[Dict[str, Any]] = None) -> None:
    """Fix the shared time range up front (autoscaling many shared axes per artist is most of a page's build): the
    data, the onset and basis lines, half a bin of margin; delivery on the right."""
    ts = np.unique(t.astype(float).dropna().to_numpy())
    xs = [*ts, *([0.0] if axis == "rel_second_stage" else []),
          *([float(pol["at"])] if pol and pol.get("at", "end") != "end" and pol.get("axis", "to_delivery") == axis else [])]
    step = float(np.diff(ts).min()) if ts.size > 1 else 0.0
    pad = 0.6 * step if step else 0.5
    lo, hi = (min(xs) - pad, max(xs) + pad) if xs else (0.0, 1.0)
    axes = list(np.ravel(axes))
    axes[0].set_xlim((hi, lo) if axis == "to_delivery" else (lo, hi))
    for ax in axes:
        ax.set_autoscalex_on(False)
        _time_ticks(ax, hi - lo, step)


def _time_ticks(ax: Any, span: float, step: float) -> None:
    """A dense time grid: a major tick at the round step that gives at most 13 labels, a minor grid line at every bin
    (``step``) when that is finer, else at half the major step."""
    from matplotlib.ticker import MultipleLocator

    major = next((m for m in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 12.0, 24.0) if span / m <= 13), 48.0)
    minor = step if 0 < step < major and span / step <= 150 else major / 2
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))


def _x_time(ax: Any, axis: str) -> None:
    """Label the shared time axis; hours before delivery inverted (delivery on the right)."""
    if axis == "to_delivery" and not ax.xaxis_inverted():
        ax.invert_xaxis()
    ax.set_xlabel(AXIS_LABEL.get(axis, axis))


def _rates(f: pd.DataFrame, pid: Optional[str]) -> Dict[str, pd.DataFrame]:
    """sens/spec/fpr bin rows of one policy (every fold) with ``under`` (the bin's underpowered marker)
    and ``raw``, the count rate (tp/n_pos, fp/n_neg) drawn hollow where the rate is NaN."""
    q = f[(f["policy_id"] == pid) | (f["metric"] == "underpowered")].set_index(["fold", "t"])
    by, none = dict(tuple(q.groupby("metric", observed=True))), q.iloc[:0]
    under, out = by.get("underpowered", none)["value"].eq(1.0), {}
    for metric, cnt, n in (("sens", "tp", "n_pos"), ("spec", "fp", "n_neg"), ("fpr", "fp", "n_neg")):
        r = by.get(metric, none)
        raw = by.get(cnt, none)["value"].reindex(r.index).astype(float) / r[n].astype(float).replace(0, np.nan)
        out[metric] = r.assign(under=under.reindex(r.index, fill_value=False),
                               raw=1.0 - raw if metric == "spec" else raw).reset_index()
    return out


def _estimate_line(ax: Any, t: Any, value: Any, raw: Any, under: Any, *, color: str, ls: str, lw: float, marker: str,
                   ms: float, label: Optional[str]) -> bool:
    """One trace over the time bins, drawn as one continuous solid line, so every trace of every time-resolved figure
    reads alike and none is broken into pieces.

    Each bin's estimate is ``value`` where the bin is powered and the count ratio ``raw`` where it is underpowered
    (``under``; ``value`` is NaN there). The line runs through every bin with an estimate at full weight ``lw``; powered
    bins carry a filled marker with a thin dark edge (:data:`figstyle.EDGE`), underpowered bins a hollow marker ringed
    in the trace's colour. An interior bin without an estimate (a row with neither, or a grid step with no row) is
    joined over, so the line stays whole, and marked where the line crosses it by a hollow marker with a black ring,
    so the joined stretch never reads as data; bins before the first or after the last estimate are not drawn. The
    line carries ``label``. ``ls`` is accepted and ignored: a trace is never dashed. Nothing is drawn when no bin has
    an estimate; returns whether it drew (and so used ``label``).
    """
    fs = _seam()
    t, value, raw = (np.asarray(a, np.float64) for a in (t, value, raw))
    under = np.asarray(under, bool) & np.isfinite(raw)
    y = np.where(under, raw, value)
    keep = np.isfinite(y) & np.isfinite(t)
    if not keep.any():
        return False
    grid = np.unique(t[np.isfinite(t)])
    order = np.argsort(t[keep], kind="stable")
    t, y, under = t[keep][order], y[keep][order], under[keep][order]
    ax.plot(t, y, color=color, lw=lw, marker=marker, ms=ms, mec=fs.EDGE, mew=fs.MARKER_EDGE,
            markevery=list(np.flatnonzero(~under)), label=label, zorder=3, solid_joinstyle="round")
    if under.any():
        ax.plot(t[under], y[under], marker, ls="none", ms=ms + 0.6, mfc="white", mec=color, mew=0.9, zorder=4)
    # the bins the line is joined over: rows without an estimate, and grid steps missing between two estimates
    step = float(np.diff(grid).min()) if grid.size > 1 else 0.0
    fill = [np.arange(a + step, b - step / 2, step) for a, b in zip(t[:-1], t[1:]) if step and b - a > 1.5 * step]
    miss = np.unique(np.round(np.concatenate([grid, *fill]), 9))
    miss = miss[(miss > t[0]) & (miss < t[-1]) & ~np.isin(miss, np.round(t, 9))]
    if miss.size:
        ax.plot(miss, np.interp(miss, t, y), marker, ls="none", ms=ms + 0.6, mfc="white", mec=fs.EDGE, mew=0.9,
                zorder=4)
    return True


def _series(ax: Any, f: pd.DataFrame, c: Dict[str, Any], *, color: str, marker: str, label: str,
            fold: Optional[str] = None, ls: str = "-") -> None:
    """Pooled (or one fold's) trace (:func:`_estimate_line`: thick where powered, faint with hollow markers where
    underpowered) with its 95% band; under it the per-fold lines (thin, faded) and, with ``fold_band: minmax``,
    their envelope."""
    fs = _seam()
    if fold is None:
        per = f[f["fold"] != "pooled"]
        for _, g in per.groupby("fold"):
            g = g.sort_values("t")
            ax.plot(g["t"], g["value"].astype(float), color=color, lw=fs.LINE_THIN * 0.8, alpha=0.22)
        if c["eval"]["fold_band"] == "minmax" and per["fold"].nunique() > 1:
            e = per.assign(value=per["value"].astype(float)).groupby("t")["value"].agg(["min", "max"])
            ax.fill_between(e.index.astype(float), e["min"], e["max"], color=color, alpha=0.07, lw=0)
    p = f[f["fold"] == _one(fold)].sort_values("t")
    t, value = p["t"].astype(float), p["value"].astype(float)
    if value.notna().any():
        ax.fill_between(t, p["ci_lo"].astype(float), p["ci_hi"].astype(float), color=color, alpha=0.15, lw=0)
    none = np.full(len(p), np.nan)
    _estimate_line(ax, t, value, p["raw"].astype(float) if "raw" in p else none,
                   p["under"].to_numpy(bool) if "under" in p else np.zeros(len(p), bool), color=color, ls=ls,
                   lw=fs.LINE_EMPHASIS * 2, marker=marker, ms=fs.MARKER_SMALL, label=label)


def _unit_y(ax: Any, ticks: tuple = (0, 0.2, 0.4, 0.6, 0.8, 1)) -> None:
    """A rate axis: [0, 1] with a little room, a major tick every 0.2 and a minor grid line every 0.1."""
    from matplotlib.ticker import MultipleLocator

    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks(list(ticks))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))


def _rates_panel(ax: Any, f: pd.DataFrame, c: Dict[str, Any], pol: Dict[str, Any], fold: Optional[str]) -> None:
    """Sensitivity, specificity and FPR of policy ``pol`` over one metric type's bin rows, with its α line."""
    fs, rates = _seam(), _rates(f, pol.get("id"))
    for metric, name, color, marker in RATES:
        _series(ax, rates[metric], c, color=color, marker=marker, fold=fold, label=name)
    if not ax.lines:
        return _empty(ax)
    if pol.get("alpha"):
        ax.axhline(pol["alpha"], ls=(0, (3, 2)), color=RATES[2][2], lw=fs.LINE_HAIRLINE, alpha=0.8,
                   label=f"α = {pol['alpha']:g}")
    _unit_y(ax)
    fs.style_axes(ax)


def _n_strip(ax: Any, f: pd.DataFrame, c: Dict[str, Any], unit: str = "GUIDs", legend: bool = True) -> None:
    """Per-bin counts of each class (``n_pos`` adverse, ``n_neg`` healthy), the underpowered floor dotted."""
    fs = _seam()
    f = f.sort_values("t")
    if f.empty:
        return _empty(ax)
    col = fs.CLINICAL_CLASS_COLORS
    for y, name, color, marker in (("n_pos", "adverse (n)", col["hie"], "o"), ("n_neg", "healthy (n)", col["healthy"], "s")):
        ax.plot(f["t"].astype(float), f[y].astype(float), marker=marker, ms=2.5, lw=fs.LINE_REGULAR, color=color,
                mec=figstyle.EDGE, mew=0.3, label=name)
    ax.axhline(c["eval"]["min_bin_class_n"], ls=":", color=fs.MUTED, lw=fs.LINE_HAIRLINE)
    top = float(np.nanmax(f[["n_pos", "n_neg"]].astype(float).to_numpy(), initial=1.0))
    ax.set(ylim=(0, 1.2 * top), yticks=[0, round(top)], ylabel=f"{unit}\nper bin")
    if legend:
        _key(ax)
    fs.style_axes(ax, grid="x")


def _under_note(fig: Any, c: Dict[str, Any], *extra: str) -> None:
    """The one note line of a time-resolved figure: what a hollow marker or a gap means, then the non-empty ``extra``
    sentences (the n strip's model, :func:`_fold_phrase`). The exclusion counts behind the axes are in
    ``evaluation/tables/inclusion.csv`` (L14), not on the figure."""
    _seam().caveat_note(fig, text=" ".join(filter(None, (UNDER_BIN.format(n=c["eval"]["min_bin_class_n"]) + ".", *extra))))


def _strip_model(models: List[tuple]) -> tuple:
    """The model whose counts fill a page's n strip: the primary model, else the first."""
    return primary_model(models) or models[0]


def _metric_types(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, pid: str, split: str = "test",
                  fold: Optional[str] = None) -> Any:
    """M1 (M3-M5 on their axes): a facet grid, rows = models with online scores, columns = the three metric types
    (instantaneous, committed cumulative, committed overall): sensitivity, specificity and FPR under ``pid`` on the
    ``eval.bin_h`` grid; one shared x axis. Under the instantaneous column, the n strip (GUIDs per class in the bin)
    of the primary model."""
    pol, one, key = _policy(c, pid), _one(fold), _Once()
    d = _sel(_tr(T, level="online", axis=axis, split=split), point="bin")
    d = d[d["subgroup"].isna()]
    models = _models(d)
    if not models:
        return _all_empty()
    fig, grid, strips = _facets(len(models), len(TYPES), strip=(0,))
    _xlim(fig.axes, d["t"], axis, pol)
    for i, (m, sd) in enumerate(models):
        x = _sel(d, model_id=m, seed=sd)
        for j, mt in enumerate(TYPES):
            ax = grid[i, j]
            _rates_panel(ax, _sel(x, metric_type=mt), c, pol, fold)
            _time_lines(ax, axis, pol)
            key(ax)
        grid[i, 0].set_ylabel("rate")
        _row_label(grid[i, -1], _who(m, sd))
    for j, mt in enumerate(TYPES):
        grid[0, j].set_title(TYPE_LABEL[mt].replace("\n", " "))
        _x_time(_bottom(grid, strips, j), axis)
    sm = _strip_model(models)
    _n_strip(strips[0], _sel(d, model_id=sm[0], seed=sm[1], metric_type="instantaneous", metric="underpowered",
                             fold=one), c)
    _time_lines(strips[0], axis, pol)
    _tag(fig, f"Rates over time · policy {pid}", split=split, fold=fold)
    _under_note(fig, c, f"n strip: {_who(*sm)}.", _fold_phrase(T, c, split))
    return fig


def _segment_instantaneous(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                           fold: Optional[str] = None) -> Any:
    """M6: rows = models; left, the segment-level instantaneous rates of the primary policy (every segment in the bin,
    segment score, GUID-cluster CIs); right, the segment AUROC per bin; the n strip (segments per class) of the primary
    model under the rates."""
    fs, one, pol, key = _seam(), _one(fold), _policy(c, c["eval"]["primary_policy"]), _Once()
    d = _sel(_tr(T, level="segment", axis=axis, split=split), point="bin")
    d = d[d["subgroup"].isna()]
    models = _models(d)
    if not models:
        return _all_empty()
    fig, grid, strips = _facets(len(models), 2, strip=(0,), width=10.0)
    _xlim(fig.axes, d["t"], axis, pol)
    for (a, b), (m, sd) in zip(grid, models):
        x = _sel(d, model_id=m, seed=sd)
        _rates_panel(a, _sel(x, metric_type="instantaneous"), c, pol, fold)
        _series(b, _sel(x, metric_type="threshold_free", metric="auroc"), c, color=fs.VIOLET, marker="D",
                label="segment AUROC", fold=fold)
        if b.lines:
            b.axhline(0.5, ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE)
            _unit_y(b)
            fs.style_axes(b)
        else:
            _empty(b)
        for ax in (a, b):
            _time_lines(ax, axis, pol)
            key(ax)
        a.set(ylabel="rate")
        b.set(ylabel="AUROC")
        _row_label(b, _who(m, sd))
    grid[0, 0].set_title(f"Rates (instantaneous, policy {pol.get('id')})")
    grid[0, 1].set_title("Segment AUROC")
    sm = _strip_model(models)
    _n_strip(strips[0], _sel(d, model_id=sm[0], seed=sm[1], metric_type="instantaneous", metric="underpowered",
                             fold=one), c, unit="segments")
    _time_lines(strips[0], axis, pol)
    for j in range(2):
        _x_time(_bottom(grid, strips, j), axis)
    _tag(fig, "Segment level over time", split=split, fold=fold)
    _under_note(fig, c, f"n strip: {_who(*sm)}.", _fold_phrase(T, c, split))
    return fig


def _model_panels(models: List[tuple], width: float = 11.0) -> tuple:
    """``(fig, axes per model, strips)`` of a page with one time panel per model, wrapped into two columns, an n strip
    under each column; unused cells hidden."""
    rows, cols = _wrap(len(models))
    fig, grid, strips = _facets(rows, cols, strip=tuple(range(cols)), width=width if cols > 1 else 7.4, row_h=2.15)
    for ax in grid.flat[len(models):]:
        ax.set_visible(False)
        if rows > 1:
            grid[-2, -1].tick_params(labelbottom=True)
    return fig, list(grid.flat[:len(models)]), strips


def _auroc_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                   fold: Optional[str] = None) -> Any:
    """R7: per model, the snapshot AUROC (bin-present GUIDs), cumulative AUROC (available GUIDs, running max) and the
    snapshot pAUC vs time with bootstrap bands; the primary model's n strip under each column; one shared x axis."""
    fs, one, alpha, key = _seam(), _one(fold), primary_alpha(c["eval"]), _Once()
    d = _sel(_tr(T, level="online", axis=axis, split=split), point="bin", metric_type="threshold_free")
    d = d[d["subgroup"].isna()]
    models = _models(d)
    if not models:
        return _all_empty()
    fig, axes, strips = _model_panels(models)
    _xlim(fig.axes, d["t"], axis)
    for a, (m, sd) in zip(axes, models):
        x = _sel(d, model_id=m, seed=sd)
        for f, color, marker, label, ls in (
                (_sel(x, denominator="bin_present", metric="auroc"), fs.BLUE, "o", "snapshot AUROC", "-"),
                (_sel(x, denominator="available", metric="auroc"), fs.ORANGE, "s", "cumulative AUROC", "-"),
                (_sel(x, denominator="bin_present", metric=f"pauc@{alpha:g}"), fs.VIOLET, "D",
                 f"snapshot pAUC@{alpha:g}", "-")):
            _series(a, f, c, color=color, marker=marker, label=label, fold=fold, ls=ls)
        if a.lines:
            a.axhline(0.5, ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE)
            _unit_y(a)
            fs.style_axes(a)
        else:
            _empty(a)
        a.set(ylabel="AUROC", title=_who(m, sd))
        _time_lines(a, axis)
        key(a)
    sm = _strip_model(models)
    snap = _sel(d, model_id=sm[0], seed=sm[1], denominator="bin_present", metric="auroc", fold=one)
    for j, s in enumerate(strips):
        _n_strip(s, snap, c, legend=j == 0)
        _time_lines(s, axis)
        _x_time(s, axis)
    _tag(fig, "AUROC over time", split=split, fold=fold)
    _under_note(fig, c, f"n strip: {_who(*sm)}.", _fold_phrase(T, c, split))
    return fig


def _roc_checkpoints(T: Dict[str, Any], c: Dict[str, Any], *, kind: str, pr: bool = False, split: str = "test",
                     fold: Optional[str] = None) -> Any:
    """R2-R4 (ROC of ``kind``) and R6 (``pr``: precision vs recall of R2): one panel per checkpoint
    and end, every model overlaid (pooled thick, per-fold thin), each model's AUC (AP) listed in its colour in the
    panel corner, the model names in the figure key; the PR panels carry each model's prevalence (dashed). The figure
    title names ``kind``."""
    fs, one = _seam(), _one(fold)
    r = _sel(T["roc"], level="online", split=split)
    ats = [f"{h:g}" for h in c["eval"]["checkpoints_h"]] + ["end"]
    models, x, y = _models(r), *(("tpr", "precision") if pr else ("fpr", "tpr"))
    fig, axes = _panels(len(ats), 4, 2.3)
    for ax, at in zip(axes, ats):
        v, stats = _sel(r, variant=f"{kind}@{at}"), []
        for i, (m, sd) in enumerate(models):
            color, g = fs.LINE_PALETTE[i % len(fs.LINE_PALETTE)], _sel(v, model_id=m, seed=sd)
            if fold is None:
                for _, h in g[g["fold"] != "pooled"].groupby("fold"):
                    ax.plot(h[x], h[y], color=color, lw=fs.LINE_THIN * 0.8, alpha=0.2)
            p = g[g["fold"] == one]
            if not len(p):
                continue
            tpr, n_pos, n = p["tpr"].to_numpy(float), p["n_pos"].iloc[0], p["n_pos"].iloc[0] + p["n_neg"].iloc[0]
            score = (np.nansum(np.diff(tpr) * p["precision"].to_numpy(float)[1:]) if pr
                     else np.trapezoid(tpr, p["fpr"].to_numpy(float)))
            ax.plot(p[x], p[y], color=color, lw=fs.LINE_EMPHASIS * 1.8, label=_who(m, sd), zorder=3)
            stats.append((color, f"{score:.2f}"))
            if pr:
                ax.axhline(n_pos / n, ls=(0, (3, 3)), color=color, lw=fs.LINE_HAIRLINE)
        ax.set_title("All segments" if at == "end" else f"{at} h before delivery")
        if not ax.lines:
            _empty(ax)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            continue
        _key(ax)
        _unit_square(ax, diagonal=not pr)
        _stat_list(ax, stats, "lower left" if pr else "lower right")
        ax.set(xlabel="recall" if pr else "FPR", ylabel="precision" if pr else "sensitivity")
        fs.style_axes(ax)
    _tag(fig, f"{ROC_KIND[kind]} {'PR' if pr else 'ROC'} at checkpoints", split=split, fold=fold)
    fs.caveat_note(fig, text=f"Numbers: {'AP' if pr else 'AUC'} of each model, in its colour. {_fold_phrase(T, c, split)}"
                   .rstrip())
    return fig


def _decision_horizon(T: Dict[str, Any], c: Dict[str, Any], *, fold: Optional[str] = None, **_: Any) -> Any:
    """R8: rows = models; sensitivity and FPR of every FPR-cap policy vs decision time c*, the threshold re-selected on
    val at each c* (committed_overall basis); ● test (band: 95% CI), ◦ val (dashed)."""
    from matplotlib.lines import Line2D

    fs, key = _seam(), _Once()
    d = _sel(T["tr"], level="online", subgroup="decision_horizon", fold=_one(fold))
    models = _models(d)
    if not models:
        return _all_empty()
    fig, grid, _ = _facets(len(models), 2, width=10.0)
    for (a, b), (m, sd) in zip(grid, models):
        x = _sel(d, model_id=m, seed=sd)
        for pid in _pids(c, x["policy_id"]):
            color = _policy_color(c, pid)
            for ax, met in ((a, "sens"), (b, "fpr")):
                for sp, face in (("val", "white"), ("test", color)):
                    g = _sel(x, policy_id=pid, split=sp, metric=met).sort_values("t")
                    if sp == "test":
                        ax.fill_between(g["t"].astype(float), g["ci_lo"].astype(float), g["ci_hi"].astype(float),
                                        color=color, alpha=0.12, lw=0)
                    ax.plot(g["t"].astype(float), g["value"].astype(float), marker="o", ms=fs.MARKER_SMALL + 0.4,
                            mfc=face, mec=color if sp == "val" else fs.EDGE,
                            mew=0.8 if sp == "val" else fs.MARKER_EDGE, color=color,
                            lw=fs.LINE_THIN if sp == "val" else fs.LINE_EMPHASIS * 2, alpha=0.7 if sp == "val" else 1.0,
                            label=pid if sp == "test" else None, zorder=3)
        for ax, what in ((a, "sensitivity"), (b, "FPR")):
            if not ax.lines:
                _empty(ax)
                continue
            ax.set(ylabel=what)
            _unit_y(ax)
            fs.style_axes(ax)
        for al in sorted({_policy(c, p).get("alpha") for p in x["policy_id"].unique()} - {None}):
            b.axhline(al, ls=(0, (3, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        key(a)
        _row_label(b, _who(m, sd))
    fs.add_key(fig, [Line2D([], [], lw=fs.LINE_THIN, marker="o", mfc="white", mec=fs.MUTED, color=fs.MUTED)],
               ["validation (thin, hollow)"])
    grid[0, 0].set_title("Sensitivity")
    grid[0, 1].set_title("FPR")
    for ax in grid[-1]:
        _x_time(ax, "to_delivery")
        ax.set_xlabel("decision time (h before delivery)")
    _tag(fig, "Decision horizon", fold=fold)
    return fig


def _threshold_stability(T: Dict[str, Any], c: Dict[str, Any], *, fold: Optional[str] = None, **_: Any) -> Any:
    """T3: rows = models; (1) the per-fold thresholds per policy (dots, min-max bar); (2) with
    ``eval.bootstrap.refit_threshold``, each fold's chosen threshold (●) with the 95% interval of its refit-bootstrap
    re-selections (bar; empty when off); (3, 4) sensitivity and FPR when each FPR-cap threshold moves by ±1/±2
    validation order statistics (● test with Wilson 95% CI, ◦ val dashed)."""
    from matplotlib.lines import Line2D

    fs, thr, key = _seam(), T["thr"], _Once()
    thr = thr.reindex(columns=list(dict.fromkeys([*thr.columns, "model_id", "seed", "level", "policy_id", "threshold", "skipped"])))
    thr = thr[thr["skipped"].isna() & (thr["level"] == "guid")]
    d = _sel(T["tr"], level="guid", subgroup="threshold_perturbation", fold=_one(fold))
    rf = _sel(T["tr"], level="guid", subgroup="threshold_refit", metric="threshold")
    models = sorted(set(_models(thr)) | set(_models(d)))
    if not models:
        return _all_empty(1, 4)
    fig, grid, _ = _facets(len(models), 4, row_h=2.15, width=12.0, sharex=False)
    for row, (m, sd) in zip(grid, models):
        t = _sel(thr, model_id=m, seed=sd)
        t = t[np.isfinite(t["threshold"].astype(float))]
        pids = _pids(c, t["policy_id"])
        for i, pid in enumerate(pids):
            v = t.loc[t["policy_id"] == pid, "threshold"].astype(float)
            row[0].vlines(i, v.min(), v.max(), color=_policy_color(c, pid), lw=fs.LINE_HEAVY * 1.6, alpha=0.3)
            row[0].plot(np.full(len(v), i), v, "o", ms=4, color=_policy_color(c, pid), mec=figstyle.EDGE, mew=0.5)
        b = _sel(rf, model_id=m, seed=sd)
        b = b[np.isfinite(b["value"].astype(float))]
        folds = sorted(b["fold"].astype(str).unique(), key=int)
        for i, pid in enumerate(pids):
            g = b[b["policy_id"] == pid]
            if not len(g):
                continue
            dx = 0.4 * (np.array([folds.index(f) for f in g["fold"].astype(str)]) / max(len(folds) - 1, 1) - 0.5)
            row[1].vlines(i + dx, g["ci_lo"].astype(float), g["ci_hi"].astype(float), color=_policy_color(c, pid),
                          lw=fs.LINE_REGULAR * 1.4)
            row[1].plot(i + dx, g["value"].astype(float), "o", ms=4, color=_policy_color(c, pid), mec=figstyle.EDGE, mew=0.5)
        for ax in row[:2]:
            if ax.lines or ax.collections:
                ax.set_xticks(range(len(pids)), pids, rotation=30, ha="right", fontsize=fs.FONT_SMALL)
                ax.set(ylabel="threshold (logit)")
                fs.style_axes(ax, grid="y")
            else:
                _empty(ax)
                ax.set(xticks=[], yticks=[])
        x = _sel(d, model_id=m, seed=sd)
        for pid in _pids(c, x["policy_id"]):
            color = _policy_color(c, pid)
            for ax, met in ((row[2], "sens"), (row[3], "fpr")):
                for sp, face in (("val", "white"), ("test", color)):
                    g = _sel(x, policy_id=pid, split=sp, metric=met)
                    g = g.assign(j=g["subgroup_value"].astype(str).astype(int)).sort_values("j")
                    v = g["value"].astype(float)
                    err = np.clip([v - g["ci_lo"].astype(float), g["ci_hi"].astype(float) - v], 0, None) if sp == "test" else None
                    ax.errorbar(g["j"], v, yerr=err, marker="o", ms=fs.MARKER_SMALL + 0.4, mfc=face,
                                mec=color if sp == "val" else fs.EDGE, mew=0.8 if sp == "val" else fs.MARKER_EDGE,
                                color=color, lw=fs.LINE_THIN if sp == "val" else fs.LINE_REGULAR * 1.3,
                                alpha=0.7 if sp == "val" else 1.0, elinewidth=fs.LINE_THIN,
                                label=pid if sp == "test" else None)
            row[3].axhline(_policy(c, pid).get("alpha") or np.nan, ls=(0, (3, 2)), color=color, lw=fs.LINE_HAIRLINE)
        for ax, what in ((row[2], "sensitivity"), (row[3], "FPR")):
            if not ax.lines:
                _empty(ax)
                continue
            ax.set(xlabel="threshold shift (order statistics)", ylabel=what, xticks=range(-2, 3))
            _unit_y(ax)
            fs.style_axes(ax)
        key(row[2])
        _row_label(row[3], _who(m, sd))
    fs.add_key(fig, [Line2D([], [], lw=fs.LINE_THIN, marker="o", mfc="white", mec=fs.MUTED, color=fs.MUTED)],
               ["validation (thin, hollow)"])
    for ax, title in zip(grid[0], ("Per-fold thresholds", "Refit interval", "Sensitivity under shift",
                                   "FPR under shift")):
        ax.set_title(title)
    _tag(fig, "Threshold stability", fold=fold)
    return fig


def _metric_type_comparison(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                            **_: Any) -> Any:
    """T4: every policy (whatever its basis) under the three metric types (columns) at every checkpoint and end;
    per model two rows, sensitivity then FPR, 95% CIs, the primary policy thick."""
    fs, ev, key = _seam(), c["eval"], _Once()
    d = _sel(_tr(T, level="online", axis="to_delivery", split=split), fold=_one(fold))
    d = d[d["subgroup"].isna() & d["point"].isin(["checkpoint", "end"]) & d["metric"].isin(["sens", "fpr"])
          & d["policy_id"].notna()]
    ats = [f"{h:g}" for h in ev["checkpoints_h"]] + ["end"]
    d = d.assign(x=[ats.index("end" if p == "end" else f"{t:g}") if p == "end" or f"{t:g}" in ats else -1
                    for p, t in zip(d["point"], d["t"].astype(float))])
    models = _models(d)
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(2 * len(models), 3, row_h=1.35, width=11.0)
    for i, (m, sd) in enumerate(models):
        x = _sel(d, model_id=m, seed=sd)
        for j, mt in enumerate(TYPES):
            for r, met in enumerate(("sens", "fpr")):
                ax = grid[2 * i + r, j]
                for pid in _pids(c, x["policy_id"]):
                    g = _sel(x, metric_type=mt, metric=met, policy_id=pid).sort_values("x")
                    v = g["value"].astype(float)
                    err = np.clip([v - g["ci_lo"].astype(float), g["ci_hi"].astype(float) - v], 0, None)
                    main = pid == ev["primary_policy"]
                    ax.errorbar(g["x"], v, yerr=err, marker="o", ms=3.5 if main else 2.5, color=_policy_color(c, pid),
                                elinewidth=fs.LINE_THIN, lw=fs.LINE_EMPHASIS * 2 if main else fs.LINE_THIN,
                                alpha=1.0 if main else 0.8, label=pid, zorder=3 if main else 2)
                if not ax.lines:
                    _empty(ax)
                    continue
                if met == "fpr":
                    for al in sorted({p.get("alpha") for p in ev["thresholds"]} - {None}):
                        ax.axhline(al, ls=(0, (3, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
                ax.set_xticks(range(len(ats)), [a if a == "end" else f"{a} h" for a in ats], fontsize=fs.FONT_SMALL)
                ax.set(ylabel=("sensitivity" if met == "sens" else "FPR") if j == 0 else None)
                _unit_y(ax)
                fs.style_axes(ax)
                key(ax)
        _row_label(grid[2 * i, -1], _who(m, sd))
        _row_label(grid[2 * i + 1, -1], _who(m, sd))
    for j, mt in enumerate(TYPES):
        grid[0, j].set_title(TYPE_LABEL[mt].replace("\n", " "))
        grid[-1, j].set_xlabel("checkpoint (h before delivery)")
    _tag(fig, f"Metric types at checkpoints · thick: {ev['primary_policy']}", split=split, fold=fold)
    return fig


def _alarms(T: Dict[str, Any], c: Dict[str, Any], *, what: str, split: str = "test", fold: Optional[str] = None) -> Any:
    """A2 (``lead_time``): rows = models; left, the cumulative detection curve (fraction of adverse GUIDs alarmed by
    c*); right, the lead-time histogram of true alarms, both on hours before delivery. A3 (``false_alarms``): left,
    the fraction of healthy GUIDs alarmed by c*; right, the time-to-first-false-alarm histogram. Every policy (latch
    rule); the primary thick with its band and fold lines, its median dotted on the histogram. The A1/A4 numbers
    (event sensitivity, lead time, burden) are in ``summary.md``."""
    fs, ev, one, key = _seam(), c["eval"], _one(fold), _Once()
    lead, primary = what == "lead_time", ev["primary_policy"]
    d = _sel(T["tr"], level="alarm", split=split, subgroup_value="latch")
    a = _sel(T["alarms"], split=split, rule="latch")
    if fold is None and len(a):  # pooled: each GUID once (data.shared_test_policy), as the pooled rows and legend
        u = a["model_id"].astype(str) + "|" + a["seed"].astype(str) + "|" + a["policy_id"].astype(str) + "|" + a["guid"]
        a = pool_rows(a.assign(unit=u, shared_test=u.duplicated(keep=False)), c["data"]["shared_test_policy"], split)
    a = a[(a["y"].astype(float) == (1.0 if lead else 0.0)) & a["alarmed"].astype(bool)]
    models = _models(d)
    if not models:
        return _all_empty(1, 2)
    fig, grid, _ = _facets(len(models), 2, row_h=1.85, width=10.0, sharex="col")
    col, curve = ("lead_time_h", "detection_frac") if lead else ("time_to_first_alarm_h", "false_alarm_frac")
    for (curve_ax, hist_ax), (m, sd) in zip(grid, models):
        x, aa = _sel(d, model_id=m, seed=sd), _sel(a, model_id=m, seed=sd)
        aa = aa[aa["fold"].astype(str) == one] if fold is not None else aa
        vals = aa[col].astype(float).dropna()
        bins = np.histogram_bin_edges(vals, bins=20) if len(vals) else None
        for pid in _pids(c, x["policy_id"]):
            color, main = _policy_color(c, pid), pid == primary
            g = _sel(x, policy_id=pid, metric=curve)
            g = g[g["point"].isin(["bin", "checkpoint"])].drop_duplicates(["fold", "t"])
            if main:
                _series(curve_ax, g, c, color=color, marker="o", label=pid, fold=fold)
            else:
                p = g[g["fold"] == one].sort_values("t")
                curve_ax.plot(p["t"].astype(float), p["value"].astype(float), color=color, lw=fs.LINE_REGULAR,
                              marker="o", ms=2.5, mew=0, alpha=0.85, label=pid)
            p = aa.loc[aa["policy_id"] == pid].dropna(subset=[col])
            # a fold page of a weighted evaluation: the fold's GUIDs at their §11.7 fold weight (alarms.parquet `w`; a
            # shared test GUID 1/K, 0 under `exclude`), as the engine's weighted median; pooled rows are each GUID once
            hw = (p["w"].astype(float).to_numpy() if fold is not None and "w" in p else np.ones(len(p)))
            v, hw = p[col].astype(float).to_numpy()[hw > 0], hw[hw > 0]
            if len(v):
                if main:
                    hist_ax.hist(v, bins=bins, weights=hw, histtype="stepfilled", color=color, alpha=0.18, lw=0)
                hist_ax.hist(v, bins=bins, weights=hw, histtype="step", color=color,
                             lw=fs.LINE_EMPHASIS * 2 if main else fs.LINE_THIN)
                if main:
                    hist_ax.axvline(float(_wquantile(v, 0.5, hw)), ls=":", color=color, lw=fs.LINE_REGULAR * 1.3)
        if not lead and _policy(c, primary).get("alpha"):
            curve_ax.axhline(_policy(c, primary)["alpha"], ls=(0, (3, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        for ax, ylabel in ((curve_ax, "fraction alarmed"), (hist_ax, "GUIDs")):
            if not ax.lines and not ax.patches:
                _empty(ax)
                continue
            ax.set(ylabel=ylabel)
            fs.style_axes(ax, grid="y" if ax is hist_ax else "both")
        curve_ax.set_ylim(-0.04, 1.04)
        key(curve_ax)
        _row_label(hist_ax, _who(m, sd))
    grid[0, 0].set_title("Adverse GUIDs alarmed by c*" if lead else "Healthy GUIDs alarmed by c*")
    grid[0, 1].set_title("Lead time of true alarms" if lead else "Time to first false alarm")
    _x_time(grid[-1, 0], "to_delivery")
    grid[-1, 0].set_xlabel("c* (h before delivery)")
    if lead:
        _x_time(grid[-1, 1], "to_delivery")
        grid[-1, 1].set_xlabel("lead time (h before delivery)")
    else:
        grid[-1, 1].set_xlabel("hours to first false alarm")
    _tag(fig, f"{'Lead time' if lead else 'False alarms'} · thick: {primary}", split=split, fold=fold)
    fs.caveat_note(fig, text=f"Dotted: median of {primary}. {_fold_phrase(T, c, split)}".rstrip())
    return fig


_BUILDERS = {
    COHORT_OVERVIEW: _cohort_overview, COHORT_SUBGROUPS: _cohort_subgroups, COHORT_TIME_BINS: _cohort_time_bins,
    COHORT_RANKED_LENGTHS: _cohort_ranked_lengths, COHORT_COVERAGE: _cohort_coverage, COHORT_GAPS: _cohort_gaps,
    COHORT_CLOCKS: _cohort_clocks, ROC_GUID: _roc_guid, PR_GUID: _pr_guid, THRESHOLD_DRIFT: _threshold_drift,
    ROC_CHECKPOINTS_CUMULATIVE: partial(_roc_checkpoints, kind="committed_cumulative"),
    ROC_CHECKPOINTS_OVERALL: partial(_roc_checkpoints, kind="committed_overall"),
    ROC_CHECKPOINTS_SNAPSHOT: partial(_roc_checkpoints, kind="snapshot"),
    ROC_SEGMENT: partial(_roc_guid, level="segment", variant="segment"),
    PR_CHECKPOINTS: partial(_roc_checkpoints, kind="committed_cumulative", pr=True),
    DECISION_HORIZON: _decision_horizon, THRESHOLD_STABILITY: _threshold_stability,
    METRIC_TYPE_COMPARISON: _metric_type_comparison,
    LEAD_TIME: partial(_alarms, what="lead_time"), FALSE_ALARMS: partial(_alarms, what="false_alarms"),
}


def _draw(stem: str, T: Dict[str, Any], c: Dict[str, Any], path: Path) -> None:
    """Render one registry stem: ``val/<stem>`` is the validation split, ``fold_<k>/<stem>`` one fold."""
    head, _, rest = stem.partition("/")
    kw = {"split": "val"} if head == "val" else {"fold": head[len("fold_"):]} if head.startswith("fold_") else {}
    _render(_figure_set(c)[rest if kw else stem](T, c, **kw), path)


def _render(fig: Any, path: Path) -> None:
    """Save ``fig`` at the stem ``path`` through :func:`figstyle.render_figure` (layout, key, title, note; closes it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _seam().render_figure(fig, path)


def load_tables(run_dir: Path) -> Dict[str, Any]:
    """Every table a figure or ``summary.md`` reads; missing ones are empty. ``metrics`` holds the P2
    rows (``point == 'n/a'``), ``tr`` the time-resolved ones (:data:`TR_COLS` only)."""
    ev, co = run_dir / "evaluation", run_dir / "cohort"

    def rd(p: Path, cols: Optional[List[str]] = None, **kw: Any) -> pd.DataFrame:
        df = pd.read_parquet(p, **kw) if p.is_file() else pd.DataFrame()
        return df if cols is None else df.reindex(columns=cols)

    mp, inc = ev / "tables" / "metrics.parquet", ev / "tables" / "inclusion.csv"
    T = {"metrics": rd(mp, filters=[("point", "==", "n/a")]),
         "tr": rd(mp, TR_COLS, columns=TR_COLS, filters=[("point", "!=", "n/a")]),
         "roc": rd(ev / "tables" / "roc_points.parquet"), "alarms": rd(ev / "tables" / "alarms.parquet", ALARM_COLS),
         "incl": (pd.read_csv(inc, dtype={"fold": str, "axis": str}) if inc.is_file()
                  else pd.DataFrame()).reindex(columns=INCL_COLS),
         "thr": rd(ev / "tables" / "thresholds.parquet"), "guids": rd(run_dir / "predictions" / "guids.parquet"),
         "fold_summary": pd.read_csv(co / "fold_summary.csv") if (co / "fold_summary.csv").is_file() else pd.DataFrame(),
         "summary": _read_json(ev / "summary.json") or {}, "manifest": _read_json(run_dir / "manifest.json") or {},
         "prov": _read_json(run_dir / "predictions" / "provenance.json") or {},
         # neural units' val calibration, keyed "fold_<k>/seed_<s>[/<kind>]" (summary.md §6); a seed ensemble's
         # folds/fold_<k>/ens[/<kind>]/ (run.ens_dir) is keyed as seed "ens"
         "calibration": {"/".join(("seed_ens" if s == "ens" else s) for s in p.parent.relative_to(run_dir / "folds").parts):
                         _read_json(p) for p in sorted(run_dir.glob("folds/fold_*/**/calibration.json"))}}
    # every figure filters these rows; categorical codes compare far faster than Arrow strings at ~1.7 M rows/model
    labels = [k for k in TR_COLS if k not in ("t", "value", "ci_lo", "ci_hi", "n_pos", "n_neg")]
    T["tr"][labels] = T["tr"][labels].astype("category")
    ovr = T["thr"]["class"].notna() if "class" in T["thr"] else pd.Series(False, index=T["thr"].index)
    T["thr"], T["thr_ovr"] = T["thr"][~ovr], T["thr"][ovr]  # T5 per-class OvR rows (3-class) apart from the binary
    have = (co / "segments.parquet").is_file() and (co / "guids.parquet").is_file()
    T["allseg"], T["seg"], T["gd"] = unique_cohort(run_dir) if have else (pd.DataFrame(),) * 3
    T |= {name: rd(ev / "tables" / f"{name}.parquet") for name in EXTRA_TABLES}
    return T


# ---- summary.md -------------------------------------------------------------------------------
def _cell(v: Any) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.3f}" if np.isfinite(v) else "-"
    return "-" if v is None else str(v)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_"
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),
                      *("| " + " | ".join(map(_cell, r)) + " |" for r in df.itertuples(index=False))])


def _ci(v: Any, ci: Any) -> str:
    ci = ci if ci and any(_cell(x) != "-" for x in ci) else None
    return "-" if v is None else _cell(float(v)) + (f" [{_cell(ci[0])}, {_cell(ci[1])}]" if ci else "")


def _value(v: pd.DataFrame, k: str) -> str:
    """The ``value`` of row ``k`` of an indexed metrics frame; '-' when absent."""
    return _cell(float(v.at[k, "value"])) if k in v.index else "-"


def _vci(v: pd.DataFrame, k: str) -> str:
    """``value [ci_lo, ci_hi]`` of row ``k`` of an indexed metrics frame; '-' when absent."""
    return _ci(v.at[k, "value"], (v.at[k, "ci_lo"], v.at[k, "ci_hi"])) if k in v.index else "-"


def _at(point: str, t: float) -> str:
    return "end" if point == "end" else f"{t:g} h"


def _time_resolved_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§8 (M2 subset): the primary policy under the three metric types at every checkpoint and end, and
    the snapshot / cumulative AUROC there (pooled test, all stages); then the M7 sanity verdicts."""
    primary = c["eval"]["primary_policy"]
    d = _sel(_tr(T, level="online", axis="to_delivery", split="test"), fold="pooled")
    d = d[d["subgroup"].isna() & d["point"].isin(["checkpoint", "end"])]
    keys = ["model_id", "seed", "metric_type", "point", "t"]
    rates, auc = [], []
    r = d[(d["policy_id"] == primary) & d["metric"].isin(["sens", "fpr"])]
    for (m, sd, mt, pt, t), g in r.groupby(keys, sort=False, dropna=False):
        v = g.set_index("metric")
        rates.append({"model": m, "seed": sd, "metric type": mt, "at": _at(pt, t),
                      "sensitivity [95% CI]": _vci(v, "sens"), "FPR [95% CI]": _vci(v, "fpr"),
                      "n+ / n-": f"{g['n_pos'].iloc[0]:.0f} / {g['n_neg'].iloc[0]:.0f}"})
    a = d[(d["metric_type"] == "threshold_free") & (d["metric"] == "auroc")]
    for (m, sd, pt, t), g in a.groupby(["model_id", "seed", "point", "t"], sort=False, dropna=False):
        v = g.set_index("denominator")
        auc.append({"model": m, "seed": sd, "at": _at(pt, t), **{
            f"{name} AUROC [CI] (n+/n-)": f"{_vci(v, k)} ({v.at[k, 'n_pos']:.0f}/{v.at[k, 'n_neg']:.0f})" if k in v.index else "-"
            for k, name in (("bin_present", "snapshot"), ("available", "cumulative"))}})
    checks = ((T["summary"].get("results") or {}).get("sanity") or {}).get("checks") or {}
    return [f"## 8. Time-resolved (primary policy `{primary}`, pooled test, hours before delivery)", "",
            "Instantaneous: GUIDs with a snapshot within the staleness window, raw score. Committed cumulative: GUIDs "
            "monitored by c*, latched. Committed overall: all GUIDs, latched (the FPR a unit would experience). "
            "Never averaged over bins (A15).", "", _md(pd.DataFrame(rates)), "",
            "Snapshot (instantaneous population) and cumulative (available population, running max) AUROC:", "",
            _md(pd.DataFrame(auc)), "", "M7 consistency checks:", "",
            *(f"- {k}: **{v.get('verdict')}** ({v.get('detail')})" for k, v in checks.items() if k.startswith("m7_")),
            "" if any(k.startswith("m7_") for k in checks) else "- not run", ""]


def _alarms_md(T: Dict[str, Any]) -> List[str]:
    """§10 part (A1-A5): event sensitivity, lead time, false alarms, burden and alarms per detection per policy."""
    d = _sel(T["tr"], level="alarm", point="end", split="test", fold="pooled")
    rows = []
    for (m, sd, rule, pid), g in d.groupby(["model_id", "seed", "subgroup_value", "policy_id"], sort=False):
        v, q = g.set_index("metric"), g.set_index("metric")["value"].astype(float)
        iqr = {k: f"{_cell(q.get(f'{k}_median_h'))} (IQR {_cell(q.get(f'{k}_q25_h'))}-{_cell(q.get(f'{k}_q75_h'))})"
               for k in ("lead_time", "ttfa")}
        rows.append({"model": m, "seed": sd, "rule": rule, "policy": pid, "event sens [CI]": _vci(v, "event_sens"),
                     "lead time h, median (IQR)": iqr["lead_time"], "healthy alarmed [CI]": _vci(v, "event_fpr"),
                     "time to 1st false alarm h": iqr["ttfa"], "burden healthy": _vci(v, "burden_neg_mean"),
                     "burden adverse": _vci(v, "burden_pos_mean"), "alarms per detection": _vci(v, "alarms_per_detection"),
                     "n+ / n-": f"{g['n_pos'].iloc[0]:.0f} / {g['n_neg'].iloc[0]:.0f}"})
    return ["### Alarm and lead-time summary (A1-A5, pooled test)", "",
            "Lead time: hours before delivery of the first (latched) alarm of a true positive. Burden: the fraction of "
            "monitored segments at or after the first alarm. Alarms per detection: FP GUIDs / TP GUIDs.", "",
            _md(pd.DataFrame(rows)), ""]


#: Threshold-free calibration metrics of §6 (``metrics.threshold_free``).
CALIB_METRICS = ("calib_intercept", "calib_slope", "ece", "ici", "brier", "scaled_brier", "logloss")


def _calibration_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§6: the val calibration of each neural unit (its ``calibration.json``: method and parameters); the calibration
    metrics of the calibrated test scores per model, level and fold, then pooled, each fold with the temperature its
    unit fitted on val (:func:`_unit_calibration`); then block K's pooled test GUID view: the calibrated score next to
    the uncalibrated one (K1) and the prior-shift-corrected one (K5, with the val and test prevalences)."""
    units = pd.DataFrame([{"unit": k, **{x: v for x, v in rec.items() if not isinstance(v, (dict, list))}}
                          for k, rec in T["calibration"].items() if rec])
    d = _sel(T["metrics"], split="test", metric_type="threshold_free")
    d = d[d["metric"].isin([*CALIB_METRICS, "pi_val", "pi_test"])] if len(d) else d
    rows, view = [], []
    for (m, sd, lvl, fold), g in d[d["subgroup"].isna()].groupby(["model_id", "seed", "level", "fold"], sort=False) \
            if len(d) else ():
        v = g.drop_duplicates("metric").set_index("metric")["value"]
        rows.append({"model": m, "seed": sd, "level": lvl, "fold": fold, **{k: v.get(k) for k in CALIB_METRICS},
                     "temperature": _num3(None if fold == "pooled" else _unit_calibration(T, m, sd, fold).get("temperature"))})
    g = _sel(d, level="guid", fold="pooled")
    for (m, sd), x in g.groupby(["model_id", "seed"], sort=False) if len(g) else ():
        prev = _sel(x, subgroup="prevalence_shift", subgroup_value="prevalence").drop_duplicates("metric").set_index("metric")["value"]
        for what, sub, val in (("calibrated", None, None), ("uncalibrated (K1)", "calibration", "uncalibrated"),
                               ("prior-shift corrected (K5)", "prevalence_shift", "prior_shift")):
            v = (x[x["subgroup"].isna()] if sub is None else _sel(x, subgroup=sub, subgroup_value=val))
            v = v.drop_duplicates("metric").set_index("metric")["value"]
            if len(v):
                view.append({"model": m, "seed": sd, "score": what, **{k: v.get(k) for k in CALIB_METRICS},
                             "π val -> test": f"{_cell(prev.get('pi_val'))} -> {_cell(prev.get('pi_test'))}"
                             if sub == "prevalence_shift" else "-"})
    return ["## 6. Calibration (calibrated scores; K1, K4, K5)", "",
            f"Fit per neural unit on its val GUID scores (`calibration.method: {c['calibration']['method']}`); the "
            "baselines are uncalibrated logits. Per unit:", "", _md(units), "",
            "Test, per level and fold, then pooled: intercept 0 and slope 1 are perfect; ECE over 10 equal-mass bins; ICI "
            "from a spline smooth; the temperature is the fold unit's val fit (a `_covoff` pass shares its unit's). "
            + _fold_def(T, c, short=True), "",
            _md(pd.DataFrame(rows)), "",
            "Pooled test, GUID level: the calibrated score, the uncalibrated one, and the calibrated one moved from each "
            "fold's val prevalence to its test prevalence (logit p - logit π_val + logit π_test).", "",
            _md(pd.DataFrame(view)), ""]


#: §7 rows (block X, GUID level): the 3-class table, the X8 ordinal table, the binary collapses and X9.
X_MAIN = ("auroc_macro", *(f"auroc_ovr_c{k}" for k in range(3)), *(f"auprc_ovr_c{k}" for k in range(3)), "bal_acc",
          *(f"recall_c{k}" for k in range(3)), "macro_f1")
X_ORDINAL = ("qwk", "rps", "auroc_hand_till", "ordinal_auroc_adverse", "ordinal_auroc_hie")
X_COLLAPSES = tuple(f"{a}/{b}" for a in ("adverse_vs_healthy", "hie_vs_rest") for b in ("auroc", "auprc"))
X_X9 = tuple(f"collapse/{n}" for n in ("auroc_binary", "auroc_aux", "spearman", "disagreement", "binary_only", "aux_only"))


def _pooled_fold_val(d: pd.DataFrame, metrics: tuple) -> pd.DataFrame:
    """Per model and metric of the GUID-level rows ``d``: pooled test [95% CI], per-fold test mean ± SD, pooled val."""
    out = []
    for (m, sd), g in d.groupby(["model_id", "seed"], sort=False) if len(d) else ():
        for met in metrics:
            x = g[g["metric"] == met]
            t, v = x[x["split"] == "test"], x[x["split"] == "val"]
            pt, per, pv = t[t["fold"] == "pooled"], t.loc[t["fold"] != "pooled", "value"], v.loc[v["fold"] == "pooled", "value"]
            if len(x):
                out.append({"model": m, "seed": sd, "metric": met,
                            "pooled test [95% CI]": _vci(pt.drop_duplicates("metric").set_index("metric"), met),
                            "fold mean ± SD": f"{_cell(per.mean())} ± {_cell(per.std())} (k={per.size})",
                            "val (optimistic)": _cell(float(pv.iloc[0])) if len(pv) else "-"})
    return pd.DataFrame(out)


def _three_class_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§7 (block X): for ``three_class``, the GUID-level 3-class metrics (pooled test with CI, fold mean ± SD, val), the
    ordinal table (X8), the pooled argmax confusion (X2), the binary collapses of the 3-class output and the per-class
    OvR thresholds (T5); for a binary task "not applicable", plus the collapse consistency (X9) of a multi-task run."""
    title, lab = "## 7. 3-class (block X, GUID level)", c["labels"]
    d = _sel(T["metrics"], level="guid")
    x9 = d[d["subgroup"].isna() & d["metric"].isin(X_X9)] if len(d) else d
    # the 3-class rows only: binary policies also write bal_acc / f1 at GUID level
    d = d[d["subgroup"].isna() & (d["metric_type"] == "threshold_free") & (d["policy_id"].isna()
                                                                          | (d["policy_id"] == "argmax"))] if len(d) else d
    note = ("Pooled test: OOF over folds with a patient-cluster bootstrap 95% CI; fold mean ± SD over the per-fold test "
            "values; val optimistic (used for selection). " + _fold_def(T, c, short=True))
    if lab["task"] != "three_class":
        L = [title, "", f"Not applicable: binary task (`labels.task: {lab['task']}`).", ""]
        if _is_multi_task(c):
            L += ["### Collapse consistency (X9): binary head vs the aux 3-class collapse", "",
                  f"The aux collapse is the aux head's P(task-positive classes); at `{c['eval']['primary_policy']}` it "
                  "gets its own threshold chosen on val by that policy. Disagreement: the fraction of the policy's basis "
                  f"GUIDs whose two decisions differ. {note}", "", _md(_pooled_fold_val(x9, X_X9)), ""]
        return L
    conf = []
    for (m, sd), g in (_sel(d, split="test", fold="pooled", policy_id="argmax").groupby(["model_id", "seed"], sort=False)
                       if len(d) else ()):
        C = g.drop_duplicates("metric").set_index("metric")["value"].reindex(CONFUSION_NAMES).to_numpy(np.float64)
        n, r = _texts(C.reshape(3, 3), ".0f"), _texts(row_normalised(C.reshape(3, 3)), ".2f")
        conf += [{"model": m, "seed": sd, "true class": CLASSES[i],
                  **{f"predicted {CLASSES[j]}": f"{n[i, j]} ({r[i, j]})" for j in range(3)}} for i in range(3)]
    thr = T["thr_ovr"]
    thr = thr[thr["skipped"].isna() & (thr["level"] == "guid")] if {"skipped", "level"} <= set(thr.columns) else thr.iloc[:0]
    if len(thr):
        agg = thr.groupby(["model_id", "seed", "class", "policy_id"], sort=False).agg(
            t=("threshold", "mean"), s=("val_sens", "mean"), f=("val_fpr", "mean")).reset_index()
        agg["cell"] = [f"{_cell(t)} ({_cell(s)} / {_cell(f)})" for t, s, f in zip(agg["t"], agg["s"], agg["f"])]
        thr = agg.assign(**{"class": agg["class"].map(lambda k: CLASSES[int(k)])}).pivot_table(
            index=["model_id", "seed", "class"], columns="policy_id", values="cell", aggfunc="first", sort=False)
        thr = thr.reindex(columns=_pids(c, thr.columns)).reset_index().rename(columns={"model_id": "model"})
    return [title, "", note, "", _md(_pooled_fold_val(d, X_MAIN)), "",
            "### Ordinal metrics (X8)", "",
            "QWK and RPS (normalised by K - 1; lower is better) of the argmax / probabilities, Hand-Till (OvO) AUROC; "
            "the ordinal alarm-score AUROC (CORAL g at y >= 1 and y >= 2, one ranking) exists for an ordinal head only "
            f"(`labels.head: {lab['head']}`).", "", _md(_pooled_fold_val(d, X_ORDINAL)), "",
            "### Argmax confusion (X2, pooled test, summed over folds)", "",
            "GUIDs (fraction of the true class); the mean of the per-fold row-normalised matrices is in figure "
            "`confusion/confusion_3class`.", "", _md(pd.DataFrame(conf)), "",
            "### Binary collapses of the 3-class output", "",
            "adverse_vs_healthy scores logit(p_c1 + p_c2) (the collapsed adverse score every binary policy thresholds); "
            "hie_vs_rest scores logit p_c2.", "", _md(_pooled_fold_val(d, X_COLLAPSES)), "",
            "### One-vs-rest thresholds (T5, GUID level)", "",
            "Per class and policy, mean over folds: threshold (calibrated OvR logit) (val sensitivity / val FPR); per "
            "fold in `tables/thresholds.parquet` (`class` column). Per-class metrics vs time use the primary policy's "
            "(figures `multiclass/per_class_vs_time_<axis>`).", "", _md(thr if len(thr) else pd.DataFrame()), ""]


def _subgroups_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§9: the subgroup table (block S, §11.6: :func:`_s_table_md`)."""
    return ["## 9. Subgroups (§11.6)", "", *_s_table_md(T, c)]


def _highlights_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§10: subgroup forest highlights (S3), restricted pairs (S5), the alarm and lead-time summary (A1-A5), fold
    heterogeneity (H1) and the top errors (E1); blocks S, H and E replace their placeholders."""
    return ["## 10. Highlights: subgroups, alarms, fold heterogeneity, errors", "",
            "### Subgroup forest (S3) and restricted pairs (S5)", "", *_s_highlights_md(T, c),
            *_alarms_md(T),
            *_heterogeneity_md(T, c),
            *_errors_md(T, c)]


def _limitations_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§11, auto-filled: internal CV only, prevalence shift, shared test GUIDs, the missingness confound, and the
    NP guarantee at GUID level only (with its empirical fallbacks and the test overshoot check)."""
    R = T["summary"].get("results") or {}
    checks, ev = (R.get("sanity") or {}).get("checks") or {}, c["eval"]
    g, prev = T["guids"], {}
    if {"fold", "split", "guid", "y"} <= set(g.columns):
        u = g.drop_duplicates(["fold", "split", "guid"])
        for s, x in u.groupby("split", sort=True):  # each split's weighted fold populations (:func:`_fold_w`)
            x = x.assign(pos=(x["y"] > 0) * 1.0, w=_fold_w(T, c, x, s))
            for f, v in x.groupby("fold"):
                pi = np.average(v["pos"], weights=v["w"]) if v["w"].sum() > 0 else np.nan
                prev.setdefault(s, []).append(f"fold {f} {pi:.2f}")
    pi_ref, shared = ev.get("reference_prevalence"), ((R.get("C") or {}).get("C12") or {}).get("n_shared_test_guids")
    np_ids = [p["id"] for p in ev["thresholds"] if p.get("method") == "np_umbrella"]
    mc, npo, n_fb = checks.get("missing_confound") or {}, checks.get("np_overshoot") or {}, (R.get("T1") or {}).get("n_fallback")
    return [
        "## 11. Limitations", "",
        f"- **Internal validation only.** {len(c['run']['folds'])}-fold cross-validation on one cohort measures the "
        "procedure, not a deployable model; temporal or external validation must follow. Naive CV confidence "
        "intervals under-cover across folds (Bates 2024).",
        "- **Prevalence shift.** Adverse prevalence per fold: " + ("; ".join(f"{s}: {', '.join(v)}" for s, v in prev.items())
                                                                  or "n/a") + ". PPV and NPV are as observed on test"
        + (f", and re-weighted to π_ref = {pi_ref}." if pi_ref is not None else
           "; not re-weighted (`eval.reference_prevalence` is null).") + " " + _fold_def(T, c, short=True),
        f"- **Shared test GUIDs.** {shared if shared is not None else 'n/a'} GUID(s) sit in more than one fold's test "
        f"split. Pooled metrics use `data.shared_test_policy: {c['data']['shared_test_policy']}`; the other policy is "
        "a sensitivity row (`subgroup = shared_test_policy`).",
        f"- **Missingness confound.** {mc.get('verdict', 'not checked')}: {mc.get('detail', 'no confound record')}.",
        f"- **NP guarantee at GUID level only.** The Neyman-Pearson umbrella ({', '.join(np_ids) or 'no policy'}) "
        "bounds P(FPR > α) ≤ δ for GUID-level decisions only; segment-level thresholds are empirical (segments are "
        f"correlated). {n_fb if n_fb is not None else 'n/a'} threshold(s) fell back to empirical (too few val "
        f"negatives) and carry no guarantee. Test check: {npo.get('verdict', 'not run')} ({npo.get('detail', '-')}).",
        ""]


def _tripod_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§12: TRIPOD+AI mini-checklist (Collins et al., BMJ 2024): which items this run's outputs support, and where."""
    R = T["summary"].get("results") or {}
    C, head, checks = R.get("C") or {}, R.get("headline") or {}, (R.get("sanity") or {}).get("checks") or {}
    lab, ev, cx = c["labels"], c["eval"], c["context"]
    by_class = (C.get("C2") or {}).get("guids_by_class")
    n_ci = sum(any(x is not None for x in h.get("test_ci") or ()) for h in head.values())
    has_cal = bool(len(_sel(T["metrics"], level="guid", metric="calib_slope")))
    cal_figs = any(s.startswith("calibration/") for s in FIGURE_REGISTRY(c))
    context = [k for k, v in cx.items() if isinstance(v, dict) and v.get("enabled")]
    context += ["covariates"] if (cx.get("covariates") or {}).get("variables") else []
    policies = ", ".join(f"{p['id']} ({p['policy']}" + (f", α {p['alpha']}" if p.get("alpha") is not None else "")
                         + (f", δ {p['delta']}" if p.get("method") == "np_umbrella" else "") + ")" for p in ev["thresholds"])
    yes = lambda ok: "yes" if ok else "no"  # noqa: E731
    items = [
        ("Data flow (recordings, exclusions)", yes(len(T["fold_summary"])),
         "§2; `cohort/fold_summary.csv`, `evaluation/tables/inclusion.csv` (L14)"),
        ("Outcome definition and timing", "yes",
         f"task `{lab['task']}` (§6.3 class mapping), head `{lab['head']}`, training labels `{lab['strategy']}`, "
         f"evaluation window `{lab['eval_window']}`; the outcome is assessed at delivery"),
        ("Predictors and timing of prediction", "yes",
         f"source `{c['source']['kind']}`, stride {(C.get('C12') or {}).get('stride_s', 'n/a')} s; context "
         f"{', '.join(context) or 'none'}; a score per segment, alarm rule `{ev['alarm_rule']['kind']}`; forbidden "
         "inputs (clock to delivery, cs, bg, source file) never enter (L1)"),
        ("Sample size (events per class)", yes(by_class), f"GUIDs per class {by_class}; per fold x split in §2"),
        ("Missing data", yes(checks.get("missing_confound")),
         f"TLO / second-stage availability per class (block C); confound check "
         f"{(checks.get('missing_confound') or {}).get('verdict', 'n/a')}; `no_valid_steps` exclusions (§2)"),
        ("Analysis (pre-specified)", "yes",
         f"{len(c['run']['folds'])}-fold CV grouped by {'patient' if c['data']['patient_map'] else 'GUID (no patient map)'}"
         f"; calibration `{c['calibration']['method']}` and thresholds on val: {policies}; primary "
         f"`{ev['primary_policy']}`; bootstrap B = {ev['bootstrap']['resamples']}"),
        ("Performance with CIs", yes(n_ci), f"{n_ci} of {len(head)} headline entries with a 95% CI (§3)"),
        ("Calibration", "yes" if has_cal and cal_figs else "partial" if has_cal else "no",
         "intercept, slope, ECE, ICI (§6)" + ("" if cal_figs else "; reliability figures pending (block K)")),
    ]
    return ["## 12. TRIPOD+AI mini-checklist", "",
            "Which TRIPOD+AI items (Collins et al., BMJ 2024) this run's outputs support; the write-up reports the rest.",
            "", _md(pd.DataFrame(items, columns=["item", "supported", "where / what"])), ""]


def _vs_frozen_md(R: Dict[str, Any]) -> List[str]:
    """§4 addendum (§10.1): each ``model`` seed (and its ensemble) minus its ``frozen`` baseline at the same seed (block
    ``VF``); nothing when the run has no frozen unit."""
    t = pd.DataFrame((R.get("VF") or {}).get("table") or [])
    if t.empty:
        return []
    pid = t["policy_id"] if "policy_id" in t else pd.Series(None, index=t.index)
    t = pd.DataFrame({
        "model": t["model_id"], "seed": t["seed"],
        "Δ (model - frozen)": [m + ("" if pd.isna(p) else f"@{p}") for m, p in zip(t["metric"], pid)],
        "pooled test [95% CI]": [_ci(v, (lo, hi)) for v, lo, hi in zip(t["value"], t["ci_lo"], t["ci_hi"])],
        "model value": t["run_value"], "frozen value": t["reference_value"], "p (bootstrap)": t["p_value"],
        "p DeLong": t.get("p_delong"), "p NB": t.get("p_nb")})
    return ["### Δ vs frozen (§10.1)", "",
            "Each `model` seed (and its seed ensemble) of the online regime minus the frozen-feature head trained at the "
            "same seed, on the pooled OOF test GUIDs both scored (unpaired rows: `inclusion.csv`, analysis VF), paired "
            "patient-cluster bootstrap; AUROC of `score_final_cal`, sens/spec at the primary policy on each model's own "
            "basis population and thresholds. NB: Nadeau-Bengio over the per-fold AUROCs.", "", _md(t), ""]


def _comparator_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§3 addendum: the pre-specified healthy comparators (family ``healthy_comparator``, §11.6.1), primary model,
    GUID level: every adverse GUID against all healthy, against the blood-gas (BG+) healthy and against the no-gas
    (BG-) healthy; AUROC and the primary policy's sensitivity and specificity, pooled test with CI, per-fold mean ± SD
    and the (optimistic) validation value. Empty when the family was not evaluated."""
    S = T.get("subgroups", pd.DataFrame())
    s = _s_primary(_sel(S, analysis="S1", subgroup="healthy_comparator", level="guid")) if len(S) else S
    if s.empty:
        return []
    pid, rows = c["eval"]["primary_policy"], []
    s = s.assign(fold=s["fold"].astype(str), subgroup_value=s["subgroup_value"].astype(str))
    for mem in member_order(s["subgroup_value"].unique()):
        x = s[s["subgroup_value"] == mem]
        n = x[(x["metric"] == "n_guids") & (x["split"] == "test") & (x["fold"] == "pooled")]
        for metric, pol in (("auroc", None), ("sens", pid), ("spec", pid)):
            m = x[(x["metric"] == metric) & (x["policy_id"].isna() if pol is None else (x["policy_id"] == pol))]
            t, v = m[m["split"] == "test"], m[m["split"] == "val"]
            per = t[t["fold"] != "pooled"]["value"].astype(float)
            rows.append({"comparator": mem, "metric": metric if pol is None else f"{metric}@{pol}",
                         "n+ / n- (pooled test)": f"{n['n_pos'].iloc[0]:.0f} / {n['n_neg'].iloc[0]:.0f}" if len(n) else "-",
                         "pooled test [95% CI]": _s_cell(t[t["fold"] == "pooled"]),
                         "fold mean ± SD": f"{_cell(per.mean())} ± {_cell(per.std())} (k={per.size})" if per.size else "-",
                         "val (optimistic)": _s_cell(v[v["fold"] == "pooled"])})
    return ["### Healthy comparator (pre-specified)", "",
            "The adverse GUIDs against three healthy populations (§11.6.1 `healthy_comparator`): `all_healthy` (every "
            "healthy GUID), `bg_healthy` (healthy GUIDs with a blood gas: a gas was drawn, so the labour raised concern) "
            "and `no_bg_healthy` (no gas). Train and val healthy GUIDs are mostly BG+, the augmented test healthy GUIDs "
            "mostly BG-, so the headline above depends on which population is meant; report both. "
            + _fold_def(T, c, short=True), "", _md(pd.DataFrame(rows)), ""]


def _np_cost_md(R: Dict[str, Any], c: Dict[str, Any], thr: pd.DataFrame) -> List[str]:
    """§5 addendum: what the NP umbrella primary policy costs against the empirical cap of the same α, basis and
    point, for the primary model at GUID level (pooled test sensitivity and FPR of both), with the validation
    negatives n₀ the order statistic was taken over: the price shrinks as n₀ grows (RESEARCH, Thresholds)."""
    ev = c["eval"]
    p = next((x for x in ev["thresholds"] if x["id"] == ev["primary_policy"]), None)
    if not p or p.get("method") != "np_umbrella":
        return []
    q = next((x for x in ev["thresholds"] if x.get("policy") == "fpr_cap" and x.get("method") == "empirical"
              and x.get("alpha") == p.get("alpha") and x.get("basis") == p.get("basis") and x.get("at") == p.get("at")),
             None)
    head = R.get("headline") or {}
    pm = primary_model(sorted({(r["model_id"], r["seed"]) for r in head.values()}))
    if q is None or pm is None:
        return []

    def get(pid: str, met: str) -> Optional[float]:
        v = (head.get(f"{pm[0]}|{pm[1]}|guid|{pid}|{met}") or {}).get("test")
        return None if v is None or not np.isfinite(float(v)) else float(v)

    s_np, f_np, s_e, f_e = (get(p["id"], "sens"), get(p["id"], "fpr"), get(q["id"], "sens"), get(q["id"], "fpr"))
    if None in (s_np, s_e, f_np, f_e):
        return []
    n0 = _sel(thr, model_id=pm[0], seed=pm[1], level="guid", policy_id=p["id"]) if len(thr) else thr
    n0 = float(n0["n_neg"].mean()) if len(n0) and "n_neg" in n0 else float("nan")
    return [f"- NP cost: `{p['id']}` (δ {p.get('delta')}) against `{q['id']}` at the same α {p['alpha']}: pooled test "
            f"sensitivity {s_np:.3f} vs {s_e:.3f} ({100 * (s_np - s_e):+.1f} points) at FPR {f_np:.3f} vs {f_e:.3f}. "
            f"The umbrella's order statistic was taken over n₀ ≈ {n0:.0f} validation negatives per fold; its price "
            f"shrinks as n₀ grows (RESEARCH, Thresholds).", ""]


def summary_md(run_dir: Path, T: Dict[str, Any], c: Dict[str, Any]) -> Path:
    """``summary.md``: the §11.11 sections 1-12, in order, each under a stable ``## <n>. `` heading (verify
    criterion 12). Sections 6, 7, 9 and 10 come from one function each (:func:`_calibration_md`,
    :func:`_three_class_md`, :func:`_subgroups_md`, :func:`_highlights_md`), which blocks K, X, S, H and E fill."""
    R = T["summary"].get("results") or {}
    rc, primary = R.get("run_context") or {}, c["eval"]["primary_policy"]
    sw, ml = rc.get("software") or {}, T["manifest"].get("mlflow")
    missing = T["prov"].get("missing_units", R.get("missing_units"))
    checks = (R.get("sanity") or {}).get("checks") or {}
    L = [f"# Classifier run summary: {run_dir.name}", "",
         "## 1. Provenance", f"- config digest: `{R.get('config_digest')}`",
         f"- source fingerprint: {rc.get('source_fingerprint') or 'n/a'}",
         f"- git: {sw.get('revision') or 'n/a'} (dirty: {sw.get('working_tree_dirty')})",
         f"- environment (run): {rc.get('versions_at_run') or 'n/a'}",
         f"- environment (evaluate): {rc.get('evaluate_env') or 'n/a'}",
         *([f"- MLflow: {ml.get('status')} (parent run: {ml.get('parent_run_id') or 'n/a'}"
            + (f"; {ml['reason']})" if ml.get("reason") else ")")] if isinstance(ml, dict) else []),
         *([f"- missing units (not trained or predicted): {', '.join(missing) if missing else 'none'}"]
           if missing is not None else []),
         f"- evaluate exit code: {T['summary'].get('exit_code')} (failed steps: {T['summary'].get('failed')})", ""]

    fs = T["fold_summary"]
    if {"fold", "split", "class_code", "level", "status", "n"} <= set(fs.columns):
        fs = fs.pivot_table(index=["fold", "split", "class_code"], columns=["level", "status"], values="n",
                            aggfunc="sum", fill_value=0)
        fs.columns = [f"{a} {b}" for a, b in fs.columns]
        fs = fs.reset_index()
    L += ["## 2. Cohort flow (per fold x split x class; retained and excluded by reason)", "", _md(fs), ""]
    nv = (R.get("C") or {}).get("no_valid_steps")
    if nv:
        L += [f"After the cohort, `no_valid_steps` (a segment whose cached step_mask is all False) excluded "
              f"{nv['segments']} segment(s) and {nv['guids']} GUID(s) over all folds and splits, from every model "
              f"(per fold x split: `tables/inclusion.csv`).", ""]

    head = pd.DataFrame(list((R.get("headline") or {}).values()))
    if len(head):
        head = head[head["policy_id"].isin([primary, "threshold_free"])]
        head = pd.DataFrame({
            "model": head["model_id"], "seed": head["seed"], "level": head["level"], "policy": head["policy_id"],
            "metric": head["metric"], "pooled test [95% CI]": [_ci(v, ci) for v, ci in zip(head["test"], head["test_ci"])],
            "fold mean ± SD": [f"{_cell(a)} ± {_cell(b)} (k={k})"
                               for a, b, k in zip(head["fold_mean"], head["fold_sd"], head["n_folds"])],
            "primary": head["primary"], "val (optimistic)": head["val"]})
    L += [f"## 3. Headline (primary policy `{primary}` and threshold-free)", "",
          "Validation values are optimistic (used for selection). AUROC/pAUC: the per-fold mean ± SD is primary, "
          "pooled OOF secondary. Naive CV intervals under-cover across folds (Bates 2024).", "", _fold_def(T, c), "",
          _md(head), "",
          *_comparator_md(T, c)]

    b1 = R.get("B1") or {}
    base = pd.DataFrame([{
        "model": r["model_id"], "seed": r["seed"], "AUROC pooled test [CI]": _ci(r["auroc_test_pooled"], r["auroc_test_ci"]),
        "AUROC fold mean ± SD": f"{_cell(r['auroc_test_fold_mean'])} ± {_cell(r['auroc_test_fold_sd'])}",
        "AUROC val (optimistic)": r["auroc_val_fold_mean"],
        **{f"sens@{k}": _ci(v["value"], v["ci"]) for k, v in r["sens_test_pooled"].items()}} for r in b1.get("table", [])])
    L += ["## 4. Baselines and controls", "", _fold_def(T, c, short=True), "", _md(base), "",
          *(f"> **{w}**" for w in b1.get("warnings", [])), "" if b1.get("warnings") else "No baseline warnings.", "",
          *(f"- {name}: **{checks[k].get('verdict')}** ({checks[k].get('detail')})"
            for k, name in (("shortcut_auroc", "shortcut baseline"), ("shuffled_auroc", "shuffled-label control"))
            if k in checks), "", *_vs_frozen_md(R)]

    thr = T["thr"]
    if len(thr):
        thr = thr[thr["skipped"].isna()].groupby(["model_id", "seed", "level", "policy_id"], as_index=False).agg(
            basis=("basis", "first"), at=("at", "first"), alpha=("alpha", "first"), thr_mean=("threshold", "mean"),
            thr_sd=("threshold", "std"), thr_min=("threshold", "min"), thr_max=("threshold", "max"),
            val_sens=("val_sens", "mean"), val_fpr=("val_fpr", "mean"), folds=("fold", "size"))
        over = (R.get("T2") or {}).get("overshoot") or {}
        key = thr["model_id"] + "|" + thr["seed"] + "|" + thr["level"] + "|" + thr["policy_id"]
        thr["overshoot pooled"] = [(over.get(k) or {}).get("pooled") for k in key]
        thr["overshoot fold mean"] = [(over.get(k) or {}).get("mean") for k in key]
        thr["overshoot fold SD"] = [(over.get(k) or {}).get("sd") for k in key]
        thr["NP tolerance"] = [(over.get(k) or {}).get("tolerance") for k in key]
    npo = checks.get("np_overshoot")
    L += ["## 5. Thresholds and FPR overshoot (test FPR - α, on each policy's basis population)", "",
          "Validation sens/FPR are the selection-time values (mean over folds). The NP guarantee holds at GUID level only; "
          "NP tolerance: the 95% binomial tolerance of the pooled test FPR above α (n_neg pooled). "
          + _fold_def(T, c, short=True),
          "", _md(thr), "", *([f"- NP overshoot check: **{npo.get('verdict')}** ({npo.get('detail')})", ""] if npo else []),
          *_np_cost_md(R, c, T["thr"])]
    L += [*_calibration_md(T, c), *_three_class_md(T, c), *_time_resolved_md(T, c), *_subgroups_md(T, c),
          *_highlights_md(T, c), *_limitations_md(T, c), *_tripod_md(T, c)]
    path = run_dir / SUMMARY_MD
    path.write_text("\n".join(L), encoding="utf-8")  # α, ±, ≤: the platform default (cp1252 on Windows) refuses them
    return path


def _refresh_artifacts(ev: Path, rep: Any) -> None:
    """Rebuild ``results.artifacts`` in ``summary.json`` from the files written since evaluate started
    (``results.evaluate_started_at``), so a figure an earlier render left behind never counts, and
    record this report's outcome as ``results.report`` (exit code, failed steps) for the verify gate."""
    from teb_vae.lag_attn.eval.report import build_manifest, json_safe

    path = ev / "summary.json"
    summary = json.loads(path.read_text())
    res, failed = summary["results"], [r.name for r in rep.failed_steps]
    try:
        res["artifacts"] = build_manifest(ev, since=res.get("evaluate_started_at"))
    except Exception:
        failed.append("artifacts")
        raise
    finally:  # always this report call's outcome, never an earlier one's
        res["report"] = {"exit_code": int(bool(failed)), "n_steps": len(rep.steps), "failed": failed}
        path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))


#: Figure worker state: ``{"key": report call, "T": its tables}`` (one load per worker and report call).
_WORKER: Dict[str, Any] = {}
#: Most figure workers; ``run.report_workers`` caps them (0: render serially here). Each worker holds its own copy of the
#: tables: ~1 GB at a fifth of a real cohort, ~3-5 GB at full scale.
N_WORKERS = 4


def _render_job(key: str, run_dir: str, c: Dict[str, Any], stem: str, fmt: str) -> Optional[str]:
    """Render one stem in one format; the traceback text on failure, else None (runs in a worker)."""
    import traceback

    try:
        if _WORKER.get("key") != key:
            _seam().configure_figure_style(c["eval"]["figure_formats"][0])
            _WORKER.update(key=key, T=load_tables(Path(run_dir)))
        _seam().figures.set_figure_format(fmt)
        _draw(stem, _WORKER["T"], c, Path(run_dir) / "evaluation" / "figures" / stem)
        return None
    except Exception:  # noqa: BLE001 - returned to the parent, which records it under Report.step
        return traceback.format_exc()


def _render_all(run_dir: Path, c: Dict[str, Any]) -> List[tuple]:
    """``[(step name, traceback | None)]`` of every registry stem x format, rendered by
    ``min(N_WORKERS, run.report_workers)`` processes forked from a ``forkserver`` that preloaded this module and the seam
    (a clean parent: ``run.py`` reports in the process that trained, with torch threads live); ``run.report_workers: 0``
    renders serially here, with one copy of the tables. A pool that cannot start or breaks falls back to serial."""
    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(stem, fmt) for fmt in c["eval"]["figure_formats"] for stem in FIGURE_REGISTRY(c)]
    key = f"{run_dir}@{time.time_ns()}"  # one per call: a worker loads the tables once, a later call reloads
    args = [(key, str(run_dir), c, stem, fmt) for stem, fmt in jobs]
    workers = min(N_WORKERS, len(jobs), int(c["run"]["report_workers"]))
    if workers < 1:
        return [(f"{stem}.{fmt}", _render_job(*a)) for (stem, fmt), a in zip(jobs, args)]
    try:
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([__name__, "teb_vae.lag_attn_cfs.eval.figures_seam"])
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            tbs = list(pool.map(_render_job, *zip(*args)))
    except Exception as e:  # noqa: BLE001 - e.g. BrokenProcessPool, or no subprocesses allowed
        logger.warning(f"parallel figure rendering failed ({e!r}); rendering serially")
        tbs = [_render_job(*a) for a in args]
    return [(f"{stem}.{fmt}", tb) for (stem, fmt), tb in zip(jobs, tbs)]


def _raise(tb: Optional[str]) -> None:
    if tb:
        raise RuntimeError(f"figure rendering failed:\n{tb}")


def report(run_dir: Any, cfg: Any) -> int:
    """Render every :func:`FIGURE_REGISTRY` stem in every ``eval.figure_formats`` (in parallel,
    :func:`_render_all`) and write ``summary.md``.

    Each figure is a fail-soft ``Report.step`` (a worker's traceback is re-raised there). Returns 1 if
    any step raised, else 0.
    """
    from teb_vae.lag_attn.eval.report import Report

    run_dir, c = Path(run_dir), classifier_cfg(cfg)
    ev, T = run_dir / "evaluation", load_tables(Path(run_dir))
    rep = Report()
    for name, tb in _render_all(run_dir, c):
        rep.step(name, _raise, tb)
    rep.step("pages", render_pages, run_dir, T, c)  # E2, block E
    rep.step(SUMMARY_MD, summary_md, run_dir, T, c)
    if (ev / "summary.json").is_file():  # the manifest's figure subset follows the active format, as before
        _seam().figures.set_figure_format(c["eval"]["figure_formats"][-1])
        rep.step("artifacts", _refresh_artifacts, ev, rep)
    for record in rep.failed_steps:
        logger.error(f"report step {record.name} failed:\n{record.traceback}")
    return rep.exit_code()


# ==== P6 blocks. Each section below belongs to one block; register figures with ``_BUILDERS`` / ``AXIS_BUILDERS`` / ``EXPECTED_WHEN`` / ``CORE_EXTRA`` / ``EXTRA_TABLES`` at its end. ====
# ---- block S (subgroups: S1-S7, R11, K3, X10) ----
# Every figure reads the block S tables: ``subgroups.parquet`` (``T["subgroups"]``: S1 counts and rates, S4 deltas, S5
# pair AUROCs, K3 reliability points), the S2/X10 rows of the time-resolved table (``T["tr"]`` with ``subgroup`` set)
# and the R11 curves in ``roc_points``. They draw the primary model (``metrics.primary_model``) and, for rates, the
# primary policy; members run worst cohort first (``metrics.member_order``) in the §11.6.2 colours (:func:`_member_style`).
# §9 (:func:`_subgroups_md`) and the S3/S5 part of §10 of summary.md read the same rows.
from teb_vae.classifier.metrics import (  # noqa: E402
    BASELINES, CLASSES, EMPTY_FAMILIES, K3_FAMILIES, RESTRICTED, ROC_FAMILIES, S2_FAMILIES, SINGLE_CLASS_FAMILIES,
    TERTILE_FAMILIES, member_order, primary_model, subgroup_families,
)

SUBGROUPS_VS_TIME = "subgroups/subgroups_vs_time_{axis}_{policy}"
SUBGROUPS_TERTILES_VS_TIME = "subgroups/subgroups_tertiles_vs_time_{axis}_{policy}"
SUBGROUP_FOREST = "subgroups/subgroup_forest"
SUBGROUP_DELTA = "subgroups/subgroup_delta"
RESTRICTED_PAIRS = "subgroups/restricted_pairs_{axis}"
COVARIATE_STRATA = "subgroups/covariate_strata"
ROC_SUBGROUPS = "roc/roc_subgroups_{family}"
CALIBRATION_SUBGROUPS = "calibration/calibration_subgroups"
PER_CLASS_SUBGROUPS = "multiclass/per_class_subgroups_{axis}"
SUBGROUP_FAMILY = "subgroups/family_{family}_to_delivery"
FOLDS_SUFFIX = "_folds"  # the fold-band twin of a time-resolved subgroup figure (S2, S5, S8)
#: S8 families, one page each: the clinical classes and their CS / BG splits (the previous pipeline's diagnosis, CS,
#: BG and healthy BG x CS plots). Acidosis and HIE have no BG split: every adverse GUID is BG+ (§2.3, EMPTY_FAMILIES).
FAMILY_PAGES = ("class", "class_x_cs", "healthy_x_bg", "healthy_bg_x_cs", "cs", "bg")
#: §11.6.2 fixed member palettes, defined once: cs/bg as the previous pipeline, the stages, a sequential 3-step tertile
#: palette, unknown grey, and ``unhealthy`` (acidosis or HIE) between the two class colours. Class and shard members
#: take ``figures_seam.group_colors``; anything else the shared line palette.
MEMBER_COLORS = {"cs_pos": figstyle.BLUE, "cs_neg": figstyle.VIOLET, "bg_pos": figstyle.ORANGE, "bg_neg": figstyle.TEAL,
                 "first": "#6C8FC7", "straddle": figstyle.VIOLET, "second": figstyle.ROSE, "unknown": figstyle.FAINT,
                 "T1": "#8DB3E2", "T2": "#3F78C2", "T3": "#1D3F7A", "unhealthy": "#D9682B"}


def _member_style(family: str, value: str, members: List[str]) -> tuple:
    """(colour, linestyle) of one member (§11.6.2): class and shard colours (``group_colors``; class_x_cs and restricted
    pairs by their class, the cs- member a lighter shade of its class), the bg colour for the healthy bg cells (in
    ``healthy_bg_x_cs`` the cs- cell a lighter shade of its bg colour, so the four cells are four colours),
    :data:`MEMBER_COLORS`, else the shared line palette in member order. The line style is always solid: members are
    told apart by colour, metrics by marker (:data:`MEMBER_METRICS`)."""
    fs = _seam()
    if family in ("class", "source_file", "class_x_cs", RESTRICTED):
        key = value if family in ("class", "source_file") else value.split("_")[0]
        color = MEMBER_COLORS.get(key) or fs.group_colors([key])[key]
        return (fs.tint(color, 0.45) if family == "class_x_cs" and value.endswith("cs_neg") else color), "-"
    if family in ("healthy_x_bg", "healthy_bg_x_cs"):
        color = MEMBER_COLORS["bg_pos" if "_bg_pos" in value else "bg_neg"]
        return (fs.tint(color, 0.45) if value.endswith("cs_neg") else color), "-"
    pal = fs.LINE_PALETTE
    return MEMBER_COLORS.get(value) or (pal[members.index(value) % len(pal)] if value in members else fs.FAINT), "-"


def _grid(n_rows: int, n_cols: int, height: float = 1.85) -> tuple:
    """``n_rows`` x ``n_cols`` panels on one shared x axis (the time-resolved subgroup pages), at the windows width."""
    fig, grid, _ = _facets(n_rows, n_cols, row_h=height)
    return fig, grid


#: The rates a subgroup trace can show and their markers (the same as :data:`RATES`): sensitivity where the member
#: holds adverse GUIDs, specificity and FPR where it holds healthy ones.
MEMBER_METRICS = {"sens": "o", "spec": "s", "fpr": "^"}


def _member_lines(ax: Any, x: pd.DataFrame, family: str, members: List[str], *, suffix: str = "",
                  pick: Optional[Callable[[str], tuple]] = None, metrics: tuple = ("sens", "spec", "fpr"),
                  named: bool = False) -> None:
    """One solid line per member and rate in ``x`` (one metric type's bin rows of ``family``): each of ``metrics`` the
    member has (sensitivity where it has adverse GUIDs, specificity and FPR where it has healthy ones), or
    ``pick(member)`` of them; the member's colour, the rate's marker (:data:`MEMBER_METRICS`); names carry ``suffix``
    (X10). Underpowered points hollow at the count ratio; the first line of a member is labelled with its name when
    ``named``."""
    fs = _seam()
    if x.empty:
        return _empty(ax)
    x = x.astype({"subgroup_value": str, "metric": str})
    v = x.set_index(["subgroup_value", "t", "metric"])["value"].unstack("metric")
    cnt = x.drop_duplicates(["subgroup_value", "t"]).set_index(["subgroup_value", "t"])[["n_pos", "n_neg"]].astype(float)
    for mem in members:
        if mem not in v.index.get_level_values(0):
            continue
        g = v.loc[mem].sort_index()
        n, t = cnt.loc[mem].reindex(g.index), g.index.to_numpy(np.float64)
        color = _member_style(family, mem, members)[0]
        under = g[f"underpowered{suffix}"].eq(1.0).to_numpy() if f"underpowered{suffix}" in g else np.zeros(len(g), bool)
        label = mem if named else None
        for met in (pick(mem) if pick else metrics):
            if f"{met}{suffix}" not in g:
                continue
            hit, den = (f"tp{suffix}", "n_pos") if met == "sens" else (f"fp{suffix}", "n_neg")
            raw = np.full(len(g), np.nan)
            if hit in g:
                with np.errstate(invalid="ignore", divide="ignore"):
                    raw = (g[hit].astype(float) / n[den]).to_numpy(np.float64)
                raw = 1.0 - raw if met == "spec" else raw
            if _estimate_line(ax, t, g[f"{met}{suffix}"].astype(float), raw, under, color=color, ls="-",
                              lw=fs.LINE_EMPHASIS * 1.7, marker=MEMBER_METRICS[met], ms=fs.MARKER_SMALL, label=label):
                label = None  # a member without this metric (e.g. healthy-only: no sensitivity) labels its next line
    if not ax.lines:
        return _empty(ax)
    _unit_y(ax)
    fs.style_axes(ax)


#: The ``*_folds`` subgroup figures read the weighted per-fold counts of S2 (§11.7: a shared test GUID at weight 1/K in
#: each of its K folds): per rate, its numerator metric row (``tp``, ``fp``: the row's ``value``) and its denominator
#: column (``n_pos``, ``n_neg``); specificity is one minus the FPR ratio.
FOLD_COUNTS = {"sens": ("tp", "n_pos"), "fpr": ("fp", "n_neg"), "spec": ("fp", "n_neg")}
#: The one note line of a ``*_folds`` figure (it replaces :data:`UNDER_BIN`, whose hollow rule is the pooled one).
#: The line is the all-fold rate of the weighted fold counts: every GUID once, but a shared test GUID as the mean of its
#: K folds' decisions, where the pooled OOF rows (`first_fold`) keep fold 1's decision alone, so it can differ slightly
#: from the pooled figures (0.5 % of the specificity/FPR cells outside the band in run 2026-10-07--17-52).
FOLD_BAND_NOTE = ("Line: all folds, each GUID once (a shared test GUID: mean of its K folds' decisions); {band}: per-fold "
                  "test rates, a shared GUID at 1/K per fold. Hollow: median fold under {n} GUIDs of a class; black "
                  "ring: no estimate")
FOLD_BAND_MISSING = ("Per-fold rows unweighted (an evaluation made before 2026-10-08: shared test GUIDs in every fold): "
                     "pooled line only; re-run --stage evaluate for the fold band")


def fold_rates(per: pd.DataFrame, met: str) -> Tuple[pd.DataFrame, pd.Series]:
    r"""One rate over the folds, from one metric type's per-fold S2 rows ``per`` (``fold``, ``subgroup_value``, ``t``,
    ``metric``, ``value``, ``n_pos``, ``n_neg``) of a weighted evaluation, whose counts are weighted (:data:`FOLD_COUNTS`).

    Returns ``(summary, rates)``. ``rates``: each fold's rate per ``(subgroup_value, t, fold)``, $\mathrm{num} /
    \mathrm{den}$ of its weighted counts, for every fold with a GUID of the class in the bin ($\mathrm{den} > 0$): every
    fold counts, however thin. ``summary`` per ``(subgroup_value, t)``: ``line`` $= \sum \mathrm{num} / \sum \mathrm{den}$
    over those folds (the pooled rate: each GUID once, a shared test GUID as the mean of its K folds' decisions),
    ``min``/``max``/``count`` of the fold rates and ``n_median``, the median weighted fold denominator. ``line`` is a
    weighted mean of the fold rates, so it lies within ``[min, max]``. Both empty when ``per`` has no numerator rows for
    ``met``."""
    num, den = FOLD_COUNTS[met]
    empty = (pd.DataFrame(columns=["line", "min", "max", "count", "n_median"]), pd.Series(dtype=float))
    per = per[per["metric"].astype(str) == num]
    if per.empty:
        return empty
    per = per.astype({"fold": str, "subgroup_value": str})
    v = per.set_index(["subgroup_value", "t", "fold"])[["value", den]].astype(float).rename(columns={"value": num})
    v = v[v[den] > 0].sort_index()
    rate = v[num] / v[den]
    g = v.groupby(level=["subgroup_value", "t"])
    line = g[num].sum() / g[den].sum()
    if met == "spec":
        rate, line = 1.0 - rate, 1.0 - line
    by = rate.groupby(level=["subgroup_value", "t"])
    summary = pd.DataFrame({"line": line, "min": by.min(), "max": by.max(), "count": by.count(),
                            "n_median": g[den].median()})
    return summary, rate


def _member_fold_lines(ax: Any, x: pd.DataFrame, family: str, members: List[str], *,
                       pick: Optional[Callable[[str], tuple]] = None, metrics: tuple = ("sens", "spec", "fpr"),
                       named: bool = False, min_n: int = 10, thin: bool = True) -> None:
    """Per member and rate (as :func:`_member_lines` picks them) of one metric type's rows ``x``: each fold's rate as a
    thin faint line (``thin``; off on the S2 pages, where up to nine traces share a panel and 90 fold lines would bury
    the bands), their min–max as a band (where at least two folds have a rate), and the pooled rate as the solid line
    over them (:func:`fold_rates`). A point is hollow where the median fold has fewer than ``min_n`` GUIDs of the
    rate's class. Without per-fold rows (an unweighted evaluation, whose per-fold rows :func:`_s_rows` drops), the
    pooled rows of ``x`` alone, as :func:`_member_lines` draws them."""
    fs = _seam()
    folds = x["fold"].astype(str)
    chosen = {mem: (pick(mem) if pick else metrics) for mem in members}
    fr = {met: fold_rates(x[folds != "pooled"], met) for met in sorted({m for v in chosen.values() for m in v})}
    if not any(len(summary) for summary, _ in fr.values()):
        return _member_lines(ax, x[folds == "pooled"], family, members, pick=pick, metrics=metrics, named=named)
    for mem in members:
        color, label = _member_style(family, mem, members)[0], (mem if named else None)
        for met in chosen[mem]:
            summary, rates = fr[met]
            if mem not in summary.index.get_level_values(0):
                continue
            sm = summary.loc[mem].sort_index()
            for _, f in (rates.loc[mem].groupby(level="fold") if thin else ()):
                f = f.droplevel("fold").sort_index()
                ax.plot(f.index.to_numpy(np.float64), f.to_numpy(np.float64), color=color, lw=fs.LINE_THIN * 0.8,
                        alpha=0.3, zorder=2)
            band = sm[sm["count"] >= 2]
            if len(band):
                ax.fill_between(band.index.to_numpy(np.float64), band["min"].astype(float), band["max"].astype(float),
                                color=color, alpha=0.12, lw=0, zorder=1)
            t, line = sm.index.to_numpy(np.float64), sm["line"].to_numpy(np.float64)
            under = sm["n_median"].to_numpy(np.float64) < min_n
            if _estimate_line(ax, t, np.where(under, np.nan, line), line, under, color=color, ls="-",
                              lw=fs.LINE_EMPHASIS * 1.7, marker=MEMBER_METRICS[met], ms=fs.MARKER_SMALL, label=label):
                label = None
    if not ax.lines:
        return _empty(ax)
    _unit_y(ax)
    fs.style_axes(ax)


def _member_key(fig: Any, metrics: tuple = ("sens", "spec", "fpr")) -> None:
    """The marker entries of a subgroup page's figure key: which rate each marker shape is."""
    from matplotlib.lines import Line2D

    fs, names = _seam(), {"sens": "sensitivity", "spec": "specificity", "fpr": "FPR"}
    fs.add_key(fig, [Line2D([], [], color=fs.MUTED, marker=MEMBER_METRICS[m], ms=4.5, mec=fs.EDGE, mew=fs.MARKER_EDGE)
                     for m in metrics], [names[m] for m in metrics])


def _member_legend(ax: Any, handles: List[Any], labels: List[str], title: Optional[str] = None) -> None:
    """The member legend right of ``ax``: one colour swatch (a line, no marker) per member."""
    from matplotlib.lines import Line2D

    fs = _seam()
    if handles:
        ax.legend([Line2D([], [], color=h.get_color(), lw=2.2) for h in handles], labels, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), fontsize=fs.FONT_SMALL, title=title, alignment="left",
                  title_fontproperties={"weight": "semibold", "size": fs.FONT_SMALL})


def _s_primary(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame``'s rows of its primary model (``metrics.primary_model``)."""
    pm = primary_model(_models(frame))
    return frame.iloc[:0] if pm is None else _sel(frame, model_id=pm[0], seed=pm[1])


def _s_rows(T: Dict[str, Any], c: Dict[str, Any], axis: str, split: str, fold: Optional[str], ovr: bool,
            all_folds: bool = False) -> pd.DataFrame:
    """The primary model's S2 (``ovr`` False) or X10 bin rows of one axis, split and fold, primary policy;
    ``all_folds``: the pooled rows and every fold's (the ``*_folds`` figures), the per-fold ones only from a weighted
    evaluation (:func:`_fold_weighted`): an older one's per-fold counts hold every shared test GUID in every fold, so
    a band from them would not frame the pooled line."""
    d = _sel(_tr(T, level="online", axis=axis, split=split), point="bin", policy_id=c["eval"]["primary_policy"])
    d = d if all_folds and _fold_weighted(T) else _sel(d, fold=_one(fold))
    return _s_primary(d[d["subgroup"].notna() & (d["metric"].astype(str).str.contains("_ovr_c") == ovr)])


def _has_fold_counts(d: pd.DataFrame) -> bool:
    """Whether ``d`` carries the weighted per-fold S2 rows (:data:`FOLD_COUNTS`; :func:`_s_rows` keeps per-fold rows
    only from a weighted evaluation)."""
    return bool(len(d)) and bool((d["fold"].astype(str) != "pooled").any())


def _folds_head(d: pd.DataFrame, folds: bool) -> str:
    """The title suffix of a ``*_folds`` figure: how many folds the band spans."""
    if not folds:
        return ""
    k = d.loc[d["fold"].astype(str) != "pooled", "fold"].nunique() if _has_fold_counts(d) else 0
    return f" · all-fold line, per-fold min–max band over {k} folds" if k else " · pooled line (no fold band)"


def _folds_note(c: Dict[str, Any], folds: bool, d: pd.DataFrame, thin: bool = True) -> Optional[str]:
    """The note line of a ``*_folds`` figure (:data:`FOLD_BAND_NOTE`, naming the thin fold lines where ``thin`` draws
    them, or :data:`FOLD_BAND_MISSING` for an unweighted evaluation), which replaces :data:`UNDER_BIN`; None for every
    other figure."""
    if not folds:
        return None
    if not _has_fold_counts(d):
        return f"{FOLD_BAND_MISSING}."
    return f"{FOLD_BAND_NOTE.format(n=c['eval']['min_subgroup_n'], band='band and thin lines' if thin else 'band')}."


def _time_page(fig: Any, grid: Any, c: Dict[str, Any], d: pd.DataFrame, axis: str, split: str, fold: Optional[str],
               head: str, note: str = "", legend_titles: Optional[List[str]] = None,
               metrics: tuple = ("sens", "spec", "fpr"), under: Optional[str] = None) -> Any:
    """Shared x range, basis/onset lines, column titles, axis labels, the title ``head`` (primary model and policy),
    the marker key (``metrics``), the one-line note (``note``, then what a hollow marker means) and, right of each row,
    the member legend of its first panel (``legend_titles`` per row) of a subgroup page."""
    fs = _seam()
    pol = _policy(c, c["eval"]["primary_policy"])
    _xlim(grid, d["t"], axis, pol)
    for j, mt in enumerate(TYPES):
        grid[0, j].set_title(TYPE_LABEL[mt].replace("\n", " "))
        _x_time(grid[-1, j], axis)
    for ax in grid.flat:
        _time_lines(ax, axis, pol)
    _tag(fig, head, split=split, fold=fold)
    _member_key(fig, metrics)
    fs.caveat_note(fig, text=" ".join(filter(None, (
        note, under if under is not None else f"{UNDER_BIN.format(n=c['eval']['min_subgroup_n'])}."))))
    for i, row in enumerate(grid):
        _member_legend(row[-1], *row[0].get_legend_handles_labels(), title=(legend_titles or [None] * len(grid))[i])
    return fig


def _subgroups_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                       fold: Optional[str] = None, tertiles: bool = False, folds: bool = False) -> Any:
    """S2: rows = subgroup families (the clinical S2 families, or with ``tertiles`` the tertile families; val and per
    fold the class family, the §11.10 core set), columns = the three metric types, one solid line per member and rate
    under the primary policy on the bin grid: sensitivity (●) where it holds adverse GUIDs, specificity (■) and FPR
    (▲) where it holds healthy ones (§11.6); primary model. ``folds`` (pooled test, the ``*_folds`` stem): each pooled
    line over its min–max band across the folds' test sets (:func:`_member_fold_lines`, without the thin fold lines:
    the S8 pages draw those, one rate per panel)."""
    d = _s_rows(T, c, axis, split, fold, ovr=False, all_folds=folds)
    draw = partial(_member_fold_lines, min_n=c["eval"]["min_subgroup_n"], thin=False) if folds else _member_lines
    pool = TERTILE_FAMILIES if tertiles else [f for f in S2_FAMILIES if f not in TERTILE_FAMILIES]
    fams = [f for f in (pool if split == "test" and fold is None else ("class",)) if f in subgroup_families(c)]
    if d.empty or not fams:
        return _all_empty(1, 3)
    fig, grid = _grid(len(fams), 3)
    for row, fam in zip(grid, fams):
        x = _sel(d, subgroup=fam)
        members = member_order(x["subgroup_value"].astype(str).unique())
        for j, (ax, mt) in enumerate(zip(row, TYPES)):
            draw(ax, _sel(x, metric_type=mt), fam, members, named=j == 0)
        row[0].set_ylabel(fam.replace("_", " "))
    pm = primary_model(_models(d))
    return _time_page(fig, grid, c, d, axis, split, fold, f"{'Tertile subgroups' if tertiles else 'Subgroups'} over "
                      f"time · {_who(*pm)} · policy {c['eval']['primary_policy']}" + _folds_head(d, folds),
                      note=_fold_phrase(T, c, split) if fold is not None else "", under=_folds_note(c, folds, d, thin=False))


def _restricted_pairs(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                      fold: Optional[str] = None, folds: bool = False) -> Any:
    """S5: one row per restricted pair (the more severe class first), columns = the three metric types: the alarm rate
    of each subtype under the primary policy (the positive subtype's sensitivity ●, the healthy FPR ▲, shared by every
    pair; an adverse second subtype its sensitivity); the pair's pooled AUROC is the row's legend title. ``folds``: each
    all-fold line (each GUID once) over its min–max band across the folds' test sets (:func:`_member_fold_lines`)."""
    pairs = [member_order(p) for p in c["eval"].get("restricted_pairs") or []]
    d = _sel(_s_rows(T, c, axis, split, fold, ovr=False, all_folds=folds), subgroup="class")
    draw = partial(_member_fold_lines, min_n=c["eval"]["min_subgroup_n"]) if folds else _member_lines
    if d.empty or not pairs:
        return _all_empty(1, 3)
    auc = _s_primary(_sel(T["subgroups"], analysis="S5", split=split, fold=_one(fold), metric="auroc"))
    fig, grid, titles = *_grid(len(pairs), 3), []
    for row, (a, b) in zip(grid, pairs):
        x = d[d["subgroup_value"].astype(str).isin([a, b])]
        for j, (ax, mt) in enumerate(zip(row, TYPES)):
            draw(ax, _sel(x, metric_type=mt), "class", [a, b], named=j == 0,
                 pick=lambda mem: ("fpr",) if mem == "healthy" else ("sens",))
        r = _sel(auc, subgroup_value=f"{a}_vs_{b}")
        titles.append(f"AUROC {_cell(float(r['value'].iloc[0]))}" if len(r) else "")
        row[0].set_ylabel(f"{a} vs {b}\nalarm rate")
    pm = primary_model(_models(d))
    return _time_page(fig, grid, c, d, axis, split, fold, f"Restricted pairs · {_who(*pm)} · policy "
                      f"{c['eval']['primary_policy']}" + _folds_head(d, folds), under=_folds_note(c, folds, d),
                      legend_titles=titles, metrics=("sens", "fpr"))


def _subgroup_family_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, family: str, axis: str = "to_delivery",
                             split: str = "test", fold: Optional[str] = None, folds: bool = False) -> Any:
    """S8: one clinical family on its own page (the previous pipeline's per-family subgroup plots): rows = the three
    metric types (instantaneous, committed cumulative, committed overall), columns = sensitivity, specificity and FPR,
    one solid line per member under the primary policy on hours before delivery (sensitivity where the member holds
    adverse GUIDs, specificity and FPR where it holds healthy ones); primary model. The S2 rows of ``family``.
    ``folds``: each all-fold line (each GUID once) over its min–max band across the folds' test sets (:func:`_member_fold_lines`)."""
    from matplotlib.lines import Line2D

    fs = _seam()
    d = _sel(_s_rows(T, c, axis, split, fold, ovr=False, all_folds=folds), subgroup=family)
    draw = partial(_member_fold_lines, min_n=c["eval"]["min_subgroup_n"]) if folds else _member_lines
    if d.empty:
        return _all_empty(len(TYPES), 3)
    fig, grid, _ = _facets(len(TYPES), 3, row_h=1.95, width=fs.WINDOWS_FIGURE_WIDTH)
    members = member_order(d["subgroup_value"].astype(str).unique())
    for i, mt in enumerate(TYPES):
        for j, met in enumerate(MEMBER_METRICS):
            draw(grid[i, j], _sel(d, metric_type=mt), family, members, metrics=(met,))
        grid[i, 0].set_ylabel("rate")
        _row_label(grid[i, -1], TYPE_LABEL[mt].replace("\n", " "))
    for ax, name in zip(grid[0], ("Sensitivity", "Specificity", "FPR")):
        ax.set_title(name)
    pol = _policy(c, c["eval"]["primary_policy"])
    _xlim(grid, d["t"], axis, pol)
    for ax in grid[-1]:
        _x_time(ax, axis)
    for ax in grid.flat:
        _time_lines(ax, axis, pol)
        _key(ax)  # the basis / onset lines
    fs.add_key(fig, [Line2D([], [], color=_member_style(family, m, members)[0], lw=2.2) for m in members], members)
    pm = primary_model(_models(d))
    _tag(fig, f"{family.replace('_', ' ')} · {_who(*pm)} · policy {c['eval']['primary_policy']}"
         + _folds_head(d, folds), split=split, fold=fold)
    fs.caveat_note(fig, text=_folds_note(c, folds, d) or f"{UNDER_BIN.format(n=c['eval']['min_subgroup_n'])}.")
    return fig


def _s_forest(T: Dict[str, Any], c: Dict[str, Any], d: pd.DataFrame, cols: list, *, ref: Dict[str, float],
              note: str, hollow: Callable[[pd.DataFrame], pd.Series]) -> Any:
    """A forest of one population's subgroup rows ``d``: one column per ``(metric, policy_id | None, label)``, one row
    per member (families in table order, members worst first, alternate families shaded), 95% CI bars, hollow where
    ``hollow(rows)`` (drawn at ``value_raw``); a dotted reference per metric (``ref``). The title is the primary model,
    the one-line ``note`` says what hollow means."""
    import matplotlib.pyplot as plt

    fs = _seam()
    d = _s_primary(d)
    if d.empty:
        return _all_empty(1, len(cols))
    d = d.astype({"subgroup": str, "subgroup_value": str, "metric": str})
    fams = [f for f in subgroup_families(c) if f in set(d["subgroup"])]
    cells = [(f, m) for f in fams for m in member_order(d.loc[d["subgroup"] == f, "subgroup_value"].unique())]
    pos, heads, y, spans = {}, {}, 0.0, []
    for i, (f, m) in enumerate(cells):
        if not i or cells[i - 1][0] != f:
            y += 0.4 if i else 0.0
            y += 1.0
            heads[f] = y
            spans.append([f, y - 0.5, y])
        y += 1.0
        pos[(f, m)] = y
        spans[-1][2] = y + 0.5
    fig, axes = plt.subplots(1, len(cols), sharey=True, squeeze=False, layout="constrained",
                             figsize=(2.1 * len(cols) + 2.4, max(2.6, 0.115 * y + 1.2)))
    for ax, (met, pid, name) in zip(axes[0], cols):
        for k, (_, lo, hi) in enumerate(spans):
            if k % 2 == 0:
                ax.axhspan(lo, hi, color=fs.STRIP, lw=0, zorder=0)
        g = d[(d["metric"] == met) & ((d["policy_id"] == pid) if pid else d["policy_id"].isna())]
        g = g.assign(y=[pos[(f, m)] for f, m in zip(g["subgroup"], g["subgroup_value"])],
                     color=[_member_style(f, m, [])[0] for f, m in zip(g["subgroup"], g["subgroup_value"])],
                     hollow=np.asarray(hollow(g), bool))
        x = np.where(g["hollow"], g["value_raw"].astype(float), g["value"].astype(float))
        ax.hlines(g["y"], g["ci_lo"].astype(float), g["ci_hi"].astype(float), colors=g["color"], lw=fs.LINE_REGULAR * 1.3)
        ax.scatter(x, g["y"], s=18, edgecolors=np.where(g["hollow"], g["color"], fs.EDGE),
                   facecolors=np.where(g["hollow"], "white", g["color"]), linewidths=0.9, zorder=3)
        if met in ref:
            ax.axvline(ref[met], ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        ax.set(title=name, ylim=(y + 0.8, 0.3))
        ticks = sorted([*((v, f"  {m}", False) for (_, m), v in pos.items()),
                        *((v, f.replace("_", " "), True) for f, v in heads.items())])
        ax.set_yticks([v for v, _, _ in ticks], [t for _, t, _ in ticks], fontsize=fs.FONT_SMALL)
        for lab, (_, _, head) in zip(ax.get_yticklabels(), ticks):
            if head:
                lab.set(fontweight="semibold", color=fs.INK)
        ax.tick_params(axis="y", length=0)
        fs.style_axes(ax, grid="x")
        if not np.isfinite(x).any():
            _empty(ax)
    pm = primary_model(_models(d))
    _tag(fig, _who(*pm))
    fs.caveat_note(fig, text=note)
    return fig


def _subgroup_forest(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """S3 (with K3's slope): per member, AUROC and pAUC (mixed members), sensitivity, specificity and FPR at the primary
    policy (as the population reports), the calibration slope; pooled test, 95% CIs (ranking: patient-cluster bootstrap;
    rates: Wilson); hollow = underpowered."""
    ev = c["eval"]
    pid, a = ev["primary_policy"], primary_alpha(ev)
    d = _sel(T["subgroups"], analysis="S1", split="test", fold="pooled", level="guid")
    cols = [("auroc", None, "AUROC"), (f"pauc@{a:g}", None, f"pAUC@{a:g}"), ("sens", pid, f"Sensitivity ({pid})"),
            ("spec", pid, f"Specificity ({pid})"), ("fpr", pid, f"FPR ({pid})"),
            ("calib_slope", None, "Calibration slope")]
    return _s_forest(T, c, d, cols, ref={"auroc": 0.5, f"pauc@{a:g}": 0.5, "calib_slope": 1.0},
                     note=f"Hollow markers: fewer than {ev['min_subgroup_n']} GUIDs of a class.",
                     hollow=lambda g: g["underpowered"].fillna(False).astype(bool))


def _subgroup_delta(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """S4: per member, the difference to its complement (every other pooled test GUID) of AUROC and of sensitivity /
    specificity / FPR at the primary policy, paired patient-cluster bootstrap 95% CIs; filled = Holm-adjusted p < 0.05
    within the family. ΔFPR is not a table row: FPR = 1 - specificity on the same GUIDs, so it is -Δspecificity with the
    interval mirrored and the same p."""
    pid = c["eval"]["primary_policy"]
    d = _sel(T["subgroups"], analysis="S4", split="test", fold="pooled")
    spec = d[d["metric"].astype(str) == "delta_spec"] if len(d) else d
    if len(spec):
        d = pd.concat([d, spec.assign(metric="delta_fpr", value=-spec["value"].astype(float),
                                      value_raw=-spec["value_raw"].astype(float), ci_lo=-spec["ci_hi"].astype(float),
                                      ci_hi=-spec["ci_lo"].astype(float))], ignore_index=True)
    cols = [("delta_auroc", None, "ΔAUROC vs complement"), ("delta_sens", pid, f"Δsensitivity ({pid})"),
            ("delta_spec", pid, f"Δspecificity ({pid})"), ("delta_fpr", pid, f"ΔFPR ({pid})")]
    return _s_forest(T, c, d, cols, ref={m: 0.0 for m, _, _ in cols},
                     note=f"Filled: Holm p < 0.05. Hollow: not significant or n < {c['eval']['min_subgroup_n']} per class.",
                     hollow=lambda g: ~(g["p_holm"].astype(float) < 0.05).to_numpy())


def _covariate_strata(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """S7: per covariate-availability family (and has_tlo), AUROC and sensitivity at the primary policy per stratum,
    pooled test, 95% CIs, for every model with a covariates-off twin (``<kind>`` filled, ``<kind>_covoff`` hollow at the
    same colour), else every non-baseline model."""
    fs, pid = _seam(), c["eval"]["primary_policy"]
    fams = [f for f in subgroup_families(c) if f.startswith("covariate:") or f == "has_tlo"]
    if not fams:
        return _all_empty(2, 1)
    d = _sel(T["subgroups"], analysis="S1", split="test", fold="pooled", level="guid")
    models = _models(d)
    kinds = [m for m in models if (f"{m[0]}_covoff", m[1]) in models]
    show = [x for k in kinds for x in (k, (f"{k[0]}_covoff", k[1]))] or [m for m in models if m[0] not in BASELINES]
    fig, axes = _figure(2, len(fams), 2.2, width=min(2.6 * len(fams) + 1.0, fs.WINDOWS_FIGURE_WIDTH))
    bases = list(dict.fromkeys(k[0].removesuffix("_covoff") for k in show))
    for j, fam in enumerate(fams):
        x = _sel(d, subgroup=fam).astype({"subgroup_value": str})
        members = member_order(x["subgroup_value"].unique())
        for ax, met, pol, name in ((axes[0, j], "auroc", None, "AUROC"), (axes[1, j], "sens", pid, f"sensitivity ({pid})")):
            g = x[(x["metric"] == met) & ((x["policy_id"] == pol) if pol else x["policy_id"].isna())]
            for i, (m, sd) in enumerate(show):
                h = _sel(g, model_id=m, seed=sd).set_index("subgroup_value").reindex(members)
                color = fs.LINE_PALETTE[bases.index(m.removesuffix("_covoff")) % len(fs.LINE_PALETTE)]
                xs = np.arange(len(members)) + (i - (len(show) - 1) / 2) * 0.14
                v, raw = h["value"].astype(float).to_numpy(), h["value_raw"].astype(float).to_numpy()
                off, fin = m.endswith("_covoff"), np.isfinite(v)
                ax.errorbar(xs[fin], v[fin], yerr=np.clip([(v - h["ci_lo"].astype(float))[fin],
                                                           (h["ci_hi"].astype(float) - v)[fin]], 0, None),
                            fmt="s" if off else "o", ms=4.5, color=color, mfc="white" if off else color,
                            mec=color if off else "white", elinewidth=fs.LINE_REGULAR, label=_who(m, sd))
                ax.plot(xs[~fin], raw[~fin], "s" if off else "o", ms=4.5, ls="none", mfc="white", mec=color,
                        label=None if fin.any() else _who(m, sd))
            if not ax.lines:
                _empty(ax)
                continue
            ax.set_xticks(range(len(members)), members)
            ax.set(ylabel=name if j == 0 else None, xlim=(-0.7, len(members) - 0.3))
            _unit_y(ax)
            if met == "auroc":
                ax.set_title(fam.removeprefix("covariate:"))
            fs.style_axes(ax, grid="y")
            _key(ax)
    _tag(fig, "Covariate strata")
    fs.caveat_note(fig, text=f"Hollow markers: fewer than {c['eval']['min_subgroup_n']} GUIDs of a class.")
    return fig


def _roc_subgroups(T: Dict[str, Any], c: Dict[str, Any], *, family: str, **_: Any) -> Any:
    """R11: the pooled test GUID ROC of every member of ``family`` (or of every restricted pair) overlaid, primary model,
    with its patient-cluster bootstrap band where both classes reach eval.min_subgroup_n; AUC in the legend."""
    fs = _seam()
    r = _sel(T["roc"], level="guid", split="test", fold="pooled")
    r = _s_primary(r[r["variant"].astype(str).str.startswith(f"subgroup:{family}=")] if len(r) else r)
    fig, axes = _figure(1, 1, 3.6, width=4.4)
    ax = axes[0, 0]
    names = r["variant"].astype(str).str.removeprefix(f"subgroup:{family}=") if len(r) else pd.Series(dtype=str)
    members = member_order(n for n in names.unique() if not n.endswith(":band"))
    for mem in members:
        p, b = r[names == mem], r[names == f"{mem}:band"]
        color, ls = _member_style(family, mem, members)
        if len(b):
            ax.fill_between(b["fpr"].astype(float), b["tpr_lo"].astype(float), b["tpr_hi"].astype(float), color=color,
                            alpha=0.12, lw=0)
        auc = np.trapezoid(p["tpr"].to_numpy(float), p["fpr"].to_numpy(float))
        ax.plot(p["fpr"], p["tpr"], color=color, ls=ls, lw=fs.LINE_EMPHASIS * 2, label=f"{mem}  {auc:.3f}")
    if not ax.lines:
        _empty(ax)
        return fig
    _unit_square(ax)
    pm = primary_model(_models(r))
    ax.set(xlabel="FPR (1 − specificity)", ylabel="sensitivity", title=f"{family.replace('_', ' ')} · {_who(*pm)}")
    _legend(ax, loc="lower right", title="member  AUC", title_fontsize=fs.FONT_SMALL)
    fs.style_axes(ax)
    return fig


def _calibration_subgroups(T: Dict[str, Any], c: Dict[str, Any], **_: Any) -> Any:
    """K3: reliability of the calibrated GUID probability per member of cs, bg, stage_last and has_tlo (one panel per
    family, equal-mass bins with Wilson bars), pooled test, primary model; slope and intercept are the S1 rows."""
    fs = _seam()
    fams = [f for f in K3_FAMILIES if f in subgroup_families(c)]
    s = _sel(T["subgroups"], split="test", fold="pooled")
    k3 = _s_primary(_sel(s, analysis="K3"))
    fig, axes = _panels(len(fams), 4, 2.4)
    for ax, fam in zip(axes, fams):
        x = _sel(k3, subgroup=fam)
        ax.set_title(fam.replace("_", " "))
        if x.empty:  # no K3 rows (e.g. no evaluation yet): the frame may lack every column
            _empty(ax)
            continue
        x = x.astype({"subgroup_value": str})
        members = member_order(x["subgroup_value"].unique())
        for mem in members:
            g = x[x["subgroup_value"] == mem].sort_values("t")
            color = _member_style(fam, mem, members)[0]
            err = np.clip([g["value"] - g["ci_lo"], g["ci_hi"] - g["value"]], 0, None).astype(float)
            ax.errorbar(g["t"], g["value"], yerr=err, color=color, marker="o", ms=4, mec=figstyle.EDGE, mew=0.5,
                        elinewidth=fs.LINE_THIN, lw=fs.LINE_EMPHASIS * 1.6, label=mem)
        if not ax.lines:
            _empty(ax)
            continue
        _unit_square(ax)
        ax.set(xlabel="mean predicted probability", ylabel="observed fraction" if ax is axes[0] else None)
        _legend(ax, loc="upper left")
        fs.style_axes(ax)
    if not fams:
        _empty(axes[0])
    pm = primary_model(_models(k3))
    _tag(fig, "Calibration by subgroup" + (f" · {_who(*pm)}" if pm else ""))
    return fig


def _per_class_subgroups(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                         fold: Optional[str] = None) -> Any:
    """X10: rows = (one-vs-rest class k, single-class family), columns = the three metric types: per member under class
    k's primary-policy OvR threshold, its sensitivity (members holding class k, solid) or specificity (dashed); primary
    model."""
    d = _s_rows(T, c, axis, split, fold, ovr=True)
    if d.empty:
        return _all_empty(1, 3)
    rows = [(k, f) for k in range(3) for f in SINGLE_CLASS_FAMILIES
            if len(d[(d["subgroup"] == f) & d["metric"].astype(str).str.endswith(f"_ovr_c{k}")])]
    fig, grid = _grid(len(rows), 3, 1.3)
    for row, (k, fam) in zip(grid, rows):
        x = _sel(d, subgroup=fam)
        x = x[x["metric"].astype(str).str.endswith(f"_ovr_c{k}")]
        members = member_order(x["subgroup_value"].astype(str).unique())
        for j, (ax, mt) in enumerate(zip(row, TYPES)):
            _member_lines(ax, _sel(x, metric_type=mt), fam, members, suffix=f"_ovr_c{k}", named=j == 0)
        row[0].set_ylabel(f"OvR {CLASSES[k]}\n{fam.replace('_', ' ')}")
    pm = primary_model(_models(d))
    return _time_page(fig, grid, c, d, axis, split, fold, f"Per-class subgroups · {_who(*pm)} · policy "
                      f"{c['eval']['primary_policy']}")


def _s_cell(r: pd.DataFrame) -> str:
    """``value [CI]`` of one subgroup row; ``- (underpowered, est. x)`` under the power guard; '-' when absent."""
    if not len(r):
        return "-"
    r = r.iloc[0]
    if bool(r.get("underpowered")) and not np.isfinite(float(r["value"])):
        return f"- (underpowered, est. {_cell(float(r['value_raw']))})"
    return _ci(r["value"], (r["ci_lo"], r["ci_hi"]))


def _s_table_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§9 body: the primary model's S1 rows (pooled test) per member, the documented-empty families, the tertile
    cut-points and the S6 family tests."""
    ev, R = c["eval"], T["summary"].get("results") or {}
    pid, s = ev["primary_policy"], _s_primary(_sel(T["subgroups"], analysis="S1", split="test", fold="pooled", level="guid"))
    if s.empty:
        return ["_no subgroup rows (tables/subgroups.parquet)_", ""]
    s = s.astype({"subgroup": str, "subgroup_value": str, "metric": str})
    rows = []
    for fam in dict.fromkeys(s["subgroup"]):
        x = s[s["subgroup"] == fam]
        for mem in member_order(x["subgroup_value"].unique()):
            g = x[x["subgroup_value"] == mem]
            n, pol = g[g["metric"] == "n_guids"], g[g["policy_id"] == pid]
            rows.append({"family": fam, "member": mem, "population": g["population"].iloc[0],
                         "n+ / n-": f"{n['n_pos'].iloc[0]:.0f} / {n['n_neg'].iloc[0]:.0f}" if len(n) else "-",
                         "AUROC [CI]": _s_cell(g[(g["metric"] == "auroc") & g["policy_id"].isna()]),
                         f"sens@{pid} [CI]": _s_cell(pol[pol["metric"] == "sens"]),
                         f"spec@{pid} [CI]": _s_cell(pol[pol["metric"] == "spec"])})
    pm = primary_model(_models(s))
    cut = ((R.get("S1") or {}).get("plan") or {}).get("cutpoints") or {}
    tests = _s_primary(_sel(T.get("subgroup_tests", pd.DataFrame()), kind="omnibus"))
    sig = tests[tests["significant"].fillna(False).astype(bool)] if len(tests) else tests
    return [f"Primary model `{pm[0]}` (seed {pm[1]}), pooled test, GUID level; policy `{pid}` on its basis population. "
            "Healthy-only members report specificity, adverse-only members sensitivity, mixed ones both plus AUROC "
            f"(§11.6). Underpowered: fewer than {ev['min_subgroup_n']} GUIDs of a needed class (`eval.min_subgroup_n`); "
            "the value is withheld and the estimate shown. Every model, split and fold: `tables/subgroups.parquet`.", "",
            _md(pd.DataFrame(rows)), "",
            f"Documented, not computed: {'; '.join(f'`{k}` ({v})' for k, v in EMPTY_FAMILIES.items())}.",
            "Tertile cut-points (pooled test GUIDs, reused for val): "
            + ("; ".join(f"{k} {', '.join(_cell(float(q)) for q in v) or 'n/a'}" for k, v in cut.items()) or "n/a") + ".",
            f"S6 family tests (Kruskal-Wallis on the GUID score per family and class, Holm across families, α 0.05): "
            f"{len(sig)} of {len(tests)} significant"
            + (": " + "; ".join(f"{r.subgroup} ({r.stratum}) p_holm {float(r.p_holm):.2g}" for r in sig.itertuples())
               if len(sig) else "") + " (`tables/subgroup_tests.parquet`, pairwise Cliff's δ there).", ""]


def _s_highlights_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§10 S3/S5 part: the largest subgroup-vs-complement differences (S4, primary model, pooled test) and the
    restricted pairs (S5 AUROC; the positive subtype's sensitivity at every policy from the S1 class rows)."""
    d = _s_primary(_sel(T["subgroups"], analysis="S4", split="test", fold="pooled"))
    d = d[np.isfinite(d["value"].astype(float))] if len(d) else d
    top = d.assign(a=d["value"].astype(float).abs(), p=d["p_holm"].astype(float)).sort_values(["p", "a"], ascending=[True, False]) \
        .head(8) if len(d) else d
    rows = [{"family": r.subgroup, "member": r.subgroup_value, "metric": r.metric, "Δ vs complement [CI]": _ci(r.value, (r.ci_lo, r.ci_hi)),
             "p (Holm, family)": f"{float(r.p_holm):.2g}", "n+ / n-": f"{r.n_pos:.0f} / {r.n_neg:.0f}"} for r in top.itertuples()]
    s = _s_primary(_sel(T["subgroups"], split="test", fold="pooled"))
    pids = [p["id"] for p in c["eval"]["thresholds"]]
    pairs = []
    for a, b in (member_order(p) for p in c["eval"].get("restricted_pairs") or []):
        au = _sel(s, analysis="S5", subgroup_value=f"{a}_vs_{b}")
        cls = _sel(s, analysis="S1", subgroup="class", subgroup_value=a, metric="sens")
        pairs.append({"pair": f"{a} vs {b}", "AUROC [CI]": _s_cell(au),
                      "n+ / n-": f"{au['n_pos'].iloc[0]:.0f} / {au['n_neg'].iloc[0]:.0f}" if len(au) else "-",
                      **{f"positive-subtype sens@{p}": _s_cell(_sel(cls, policy_id=p)) for p in pids}})
    none = (f"_no member where both it and its complement reach eval.min_subgroup_n = {c['eval']['min_subgroup_n']} GUIDs "
            "of the needed class (estimates: `tables/subgroups.parquet`, `value_raw`)_")
    return ["The largest differences to the complement (S4: paired patient-cluster bootstrap; Holm within the family), "
            "pooled test, primary model:", "", _md(pd.DataFrame(rows)) if rows else none, "",
            "Restricted pairs (S5): AUROC of the GUID score between the two classes (the more severe positive); its "
            "sensitivity at each policy is the S1 class row, and the healthy FPR is shared by every pair.", "",
            _md(pd.DataFrame(pairs)), ""]


AXIS_BUILDERS[SUBGROUPS_VS_TIME] = _subgroups_vs_time
EXPECTED_WHEN[SUBGROUPS_VS_TIME] = lambda c: bool(set(S2_FAMILIES) & set(subgroup_families(c)))
CORE_EXTRA.append(SUBGROUPS_VS_TIME)  # val/ and fold_<k>/ draw the class family only (§11.10 core set)
AXIS_BUILDERS[SUBGROUPS_TERTILES_VS_TIME] = partial(_subgroups_vs_time, tertiles=True)  # S2, the tertile families
EXPECTED_WHEN[SUBGROUPS_TERTILES_VS_TIME] = lambda c: bool(set(TERTILE_FAMILIES) & set(subgroup_families(c)))
_BUILDERS[SUBGROUP_FOREST] = _subgroup_forest
_BUILDERS[SUBGROUP_DELTA] = _subgroup_delta
EXPECTED_WHEN[SUBGROUP_FOREST] = EXPECTED_WHEN[SUBGROUP_DELTA] = lambda c: bool(subgroup_families(c))
AXIS_BUILDERS[RESTRICTED_PAIRS] = _restricted_pairs
EXPECTED_WHEN[RESTRICTED_PAIRS] = lambda c: bool((c.get("eval") or {}).get("restricted_pairs"))
_BUILDERS[COVARIATE_STRATA] = _covariate_strata
EXPECTED_WHEN[COVARIATE_STRATA] = lambda c: any(f.startswith("covariate:") for f in subgroup_families(c))
for _f in (*ROC_FAMILIES, RESTRICTED):
    _BUILDERS[ROC_SUBGROUPS.format(family=_f)] = partial(_roc_subgroups, family=_f)
    EXPECTED_WHEN[ROC_SUBGROUPS.format(family=_f)] = EXPECTED_WHEN[RESTRICTED_PAIRS] if _f == RESTRICTED else (
        lambda c, f=_f: f in subgroup_families(c))
for _f in FAMILY_PAGES:  # S8: the previous pipeline's per-family subgroup plots, one wide page each
    _BUILDERS[SUBGROUP_FAMILY.format(family=_f)] = partial(_subgroup_family_vs_time, family=_f)
    EXPECTED_WHEN[SUBGROUP_FAMILY.format(family=_f)] = lambda c, f=_f: (
        f in subgroup_families(c) and "to_delivery" in (c.get("eval") or {}).get("time_axes", []))


def _multi_fold(c: Dict[str, Any]) -> bool:
    """A fold band needs at least two folds, and ``eval.fold_band: minmax`` (``none`` turns every fold band off)."""
    return len((c.get("run") or {}).get("folds") or []) > 1 and (c.get("eval") or {}).get("fold_band") != "none"


# The ``*_folds`` twins of the time-resolved subgroup figures (S2, S5, S8): the all-fold lines (each GUID once) over a min–max band
# across the folds' test sets (:func:`_member_fold_lines`); pooled test only, so not in CORE_EXTRA.
for _s, _b in ((SUBGROUPS_VS_TIME, AXIS_BUILDERS), (SUBGROUPS_TERTILES_VS_TIME, AXIS_BUILDERS),
               (RESTRICTED_PAIRS, AXIS_BUILDERS),
               *((SUBGROUP_FAMILY.format(family=_f), _BUILDERS) for _f in FAMILY_PAGES)):
    _b[_s + FOLDS_SUFFIX] = partial(_b[_s], folds=True)
    EXPECTED_WHEN[_s + FOLDS_SUFFIX] = lambda c, s=_s: EXPECTED_WHEN[s](c) and _multi_fold(c)
_BUILDERS[CALIBRATION_SUBGROUPS] = _calibration_subgroups
EXPECTED_WHEN[CALIBRATION_SUBGROUPS] = lambda c: bool(set(K3_FAMILIES) & set(subgroup_families(c)))
AXIS_BUILDERS[PER_CLASS_SUBGROUPS] = _per_class_subgroups
EXPECTED_WHEN[PER_CLASS_SUBGROUPS] = lambda c: ((c.get("labels") or {}).get("task") == "three_class"
                                                 and (c.get("eval") or {}).get("ovr_thresholds") in ("auto", True))
EXTRA_TABLES.extend(["subgroups", "subgroup_tests"])


# ---- block KH (calibration and heterogeneity: K1, K2, K4-K7, H1-H4, R10, R12, R13) ----
# Reliability is drawn from ``predictions/guids.parquet`` (probabilities per GUID) with the legend numbers read from the
# metrics rows (``metrics.threshold_free``); K6 from ``decision_curve.parquet``, R12/R13 from ``snapshots.parquet``, K7
# and R10 from the time-resolved rows and ``roc_points``. §6 (:func:`_calibration_md`) and the H1/H4 part of §10
# (:func:`_heterogeneity_md`) of summary.md read the same rows.
from scipy.special import expit  # noqa: E402

from teb_vae.classifier.metrics import BASELINES, CLASSES, _ece, adjust_ppv_npv, prior_shifted, wilson  # noqa: E402

CALIBRATION_GUID = "calibration/calibration_guid"
CALIBRATION_PER_CLASS = "calibration/calibration_per_class"
CALIBRATION_FOLDS = "calibration/calibration_folds"
PREVALENCE_SHIFT = "calibration/prevalence_shift"
DECISION_CURVE = "calibration/decision_curve"
BRIER_VS_TIME = "calibration/brier_vs_time_{axis}"
FOLD_FOREST = "heterogeneity/fold_forest"
SEED_SPREAD = "heterogeneity/seed_spread"
ROC_STAGE = "roc/roc_stage"
SCORE_DISTRIBUTIONS = "roc/score_distributions"
SCORE_WINDOWS = "roc/score_windows_{axis}"
P3_CAL = [f"p_c{k}_cal" for k in range(3)]


def _guid_rows(T: Dict[str, Any], c: Dict[str, Any], m: str, sd: str, split: str, fold: Optional[str],
               need: Any = ("score_final_cal",)) -> pd.DataFrame:
    """One model/seed/split's GUID predictions with finite ``need`` and their weight ``w``: one fold's (its weighted
    fold population, :func:`_fold_rows`), or pooled at weight 1 (each GUID once under ``data.shared_test_policy``,
    :func:`pool_rows`)."""
    g = _fold_rows(T, c, m, sd, split, need)
    if fold is not None:
        return g[g["fold"].astype(str) == str(fold)]
    return pool_rows(g.assign(unit=g["guid"]), c["data"]["shared_test_policy"], split).assign(w=1.0)


def _fold_rows(T: Dict[str, Any], c: Dict[str, Any], m: str, sd: str, split: str,
               need: Any = ("score_final_cal",)) -> pd.DataFrame:
    """Every fold's GUID predictions of one model/seed/split with finite ``need``, each row with its weight ``w`` in its
    fold's population (:func:`_fold_w`); group them by ``fold`` for the per-fold curves."""
    g = _sel(T["guids"], model_id=m, seed=sd, split=split)
    if not len(g) or not set(need) <= set(g.columns):
        return g.iloc[:0].assign(w=1.0)
    g = g.dropna(subset=list(need))
    return g.assign(w=_fold_w(T, c, g, split))


def _reliability(y: Any, p: Any, w: Any = None) -> pd.DataFrame:
    r"""Equal-mass reliability bins of ``p``, binned as :func:`metrics._ece` (``calibration_curve(strategy='quantile')``)
    with at most 10 bins and at least 5 GUIDs each: mean predicted ``pred``, observed fraction ``obs`` with its Wilson
    95% interval ``lo``/``hi``, and ``n``. With row weights ``w`` (a weighted fold population, :func:`_fold_w`), the
    zero-weight rows are left out, the bin edges are weighted quantiles (:func:`metrics._wquantile`, the edges
    :func:`metrics._ece` uses), ``pred``, ``k`` $= \sum w y$ and ``n`` $= \sum w$ are weighted, and the interval is
    Wilson at Kish's effective size $(\sum w)^2/\sum w^2$ (:func:`metrics.kish_wilson`). Unit weights reproduce the
    unweighted bins and intervals exactly."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    w = np.ones(p.size) if w is None else np.asarray(w, np.float64)
    y, p, w = y[w > 0], p[w > 0], w[w > 0]
    if not p.size:
        return pd.DataFrame(columns=["pred", "k", "n", "obs", "lo", "hi"])
    n_bins = int(np.clip(p.size // 5, 1, 10))
    ids = np.searchsorted(np.asarray(_wquantile(p, np.linspace(0, 1, n_bins + 1), w))[1:-1], p)
    d = pd.DataFrame({"b": ids, "wp": w * p, "wy": w * y, "w": w, "w2": w * w}).groupby("b").agg(
        wp=("wp", "sum"), k=("wy", "sum"), n=("w", "sum"), n2=("w2", "sum"))
    k, n, n2 = (d[x].to_numpy() for x in ("k", "n", "n2"))
    lo, hi = wilson(k * n / n2, n * n / n2)  # kish_wilson's form per bin; = wilson(k, n) at unit weights
    return d.assign(pred=d["wp"] / d["n"], obs=d["k"] / d["n"], lo=lo, hi=hi).drop(columns=["wp", "n2"])


def _rel_plot(ax: Any, r: pd.DataFrame, color: str, label: Optional[str] = None, *, thin: bool = False) -> None:
    """One reliability curve: thin (a fold's) or thick with Wilson bars (the population's)."""
    fs = _seam()
    if thin:
        ax.plot(r["pred"], r["obs"], color=color, lw=fs.LINE_THIN, alpha=0.3)
        return
    err = np.clip([r["obs"] - r["lo"], r["hi"] - r["obs"]], 0, None)
    ax.errorbar(r["pred"], r["obs"], yerr=err, color=color, marker="o", ms=4.2, mec=figstyle.EDGE, mew=0.5,
                elinewidth=fs.LINE_THIN, lw=fs.LINE_EMPHASIS * 2, label=label, zorder=3)


def _rel_frame(ax: Any, title: str, ylabel: Optional[str] = "observed fraction", key: Optional[_Once] = None) -> None:
    """The reliability frame: diagonal, unit square, labels and the key (``key``: once per figure, else in-panel)."""
    fs = _seam()
    _unit_square(ax)
    ax.set(ylabel=ylabel, title=title)
    (key or _legend)(ax)
    fs.style_axes(ax)


def _calib_rows(T: Dict[str, Any], m: str, sd: str, split: str, fold: Optional[str], subgroup: Optional[str] = None,
                value: Optional[str] = None) -> pd.DataFrame:
    """GUID threshold-free rows of one population: the calibrated score's (``subgroup`` None) or a tagged variant's."""
    d = _sel(T["metrics"], model_id=m, seed=sd, level="guid", split=split, fold=_one(fold), metric_type="threshold_free")
    return d[d["subgroup"].isna()] if subgroup is None else _sel(d, subgroup=subgroup, subgroup_value=value)


def _calibration_guid(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                      **_: Any) -> Any:
    """K1: per model (wrapped, three per row), the equal-mass reliability of the GUID probability after calibration
    (sigma(score_final_cal): thick with Wilson bars, per fold thin under the pooled one) and before it
    (sigma(score_final), dashed, drawn when calibration moved it); under it, the calibrated probability by outcome.
    A fold (the thin curves, a ``fold k`` page) is its weighted fold population (:func:`_fold_rows`). Slope, intercept,
    ECE and ICI are in ``summary.md`` (section 6) and K4. The keys are drawn once."""
    import matplotlib.pyplot as plt

    fs, key, hist_key = _seam(), _Once(), _Once()
    models = _models(_sel(T["guids"], split=split))
    if not models:
        return _all_empty(2, 1)
    rows, cols = _wrap(len(models), 3)
    fig, axes = plt.subplots(2 * rows, cols, squeeze=False, sharex="col", layout="constrained",
                             gridspec_kw={"height_ratios": [1.0, 0.36] * rows}, figsize=(2.6 * cols + 0.6, 3.35 * rows + 0.7))
    col = fs.CLINICAL_CLASS_COLORS
    for i in range(rows * cols):
        ax, hist = axes[2 * (i // cols), i % cols], axes[2 * (i // cols) + 1, i % cols]
        if i >= len(models):
            ax.set_visible(False), hist.set_visible(False)
            continue
        m, sd = models[i]
        g = _guid_rows(T, c, m, sd, split, fold, ("score_final", "score_final_cal"))
        ax.set_title(_who(m, sd))
        if not len(g):
            _empty(ax), _empty(hist)
            continue
        y, p, p0 = g["y"].to_numpy(np.float64), expit(g["score_final_cal"].to_numpy(np.float64)), expit(
            g["score_final"].to_numpy(np.float64))
        w = g["w"].to_numpy(np.float64)
        if fold is None:
            for _, f in _fold_rows(T, c, m, sd, split).groupby("fold"):
                _rel_plot(ax, _reliability(f["y"], expit(f["score_final_cal"].to_numpy(np.float64)), f["w"]), fs.BLUE,
                          thin=True)
        _rel_plot(ax, _reliability(y, p, w), fs.BLUE, "calibrated")
        if not np.allclose(p, p0):
            r0 = _reliability(y, p0, w)
            # filled: hollow markers mean an underpowered bin in every time-resolved figure
            ax.plot(r0["pred"], r0["obs"], "o-", ms=3.5, color=fs.ORANGE, mec=figstyle.EDGE, mew=0.4, lw=fs.LINE_REGULAR,
                    label="uncalibrated")
        _rel_frame(ax, _who(m, sd), "observed fraction adverse" if i % cols == 0 else None, key)
        ax.set_aspect("auto")
        for flag, name, color in ((0.0, "healthy", col["healthy"]), (1.0, "adverse", col["hie"])):
            if (y == flag).any():
                hist.hist(p[y == flag], bins=np.linspace(0, 1, 21), weights=w[y == flag], histtype="stepfilled",
                          color=color, alpha=0.22, lw=0)
                hist.hist(p[y == flag], bins=np.linspace(0, 1, 21), weights=w[y == flag], histtype="step", color=color,
                          lw=fs.LINE_REGULAR, label=name)
        hist.set(ylabel="GUIDs" if i % cols == 0 else None, xlabel="predicted probability")
        hist.tick_params(labelbottom=True)
        hist_key(hist)
        fs.style_axes(hist, grid="y")
    _tag(fig, "GUID calibration", split=split, fold=fold)
    _seam().caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _calibration_per_class(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                           **_: Any) -> Any:
    """K2 (3-class): rows = models, one panel per class k: the equal-mass reliability of the calibrated P(class k)
    against 1[class = k] (thick with Wilson bars, per fold thin on its weighted fold population), its ECE in the panel
    corner."""
    fs = _seam()
    g = _sel(T["guids"], split=split)
    models = _models(g.dropna(subset=P3_CAL)) if len(g) and set(P3_CAL) <= set(g.columns) else []
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=2.5, width=8.6, sharex=False)
    colors = fs.group_colors(CLASSES)
    for row, (m, sd) in zip(grid, models):
        x, per = _guid_rows(T, c, m, sd, split, fold, P3_CAL), _fold_rows(T, c, m, sd, split, P3_CAL)
        for k, (ax, name) in enumerate(zip(row, CLASSES)):
            if fold is None:
                for _, f in per.groupby("fold"):
                    _rel_plot(ax, _reliability(f["class_code"] - 1 == k, f[P3_CAL[k]], f["w"]), colors[name], thin=True)
            yk, pk = (x["class_code"].to_numpy() - 1 == k).astype(np.float64), x[P3_CAL[k]].to_numpy(np.float64)
            if yk.size:
                _rel_plot(ax, _reliability(yk, pk, x["w"]), colors[name])
                # ponytail: the ECE is unweighted, exact on the pooled page (w = 1); a fold page's would need weights
                _stat(ax, f"ECE {_ece(yk, pk):.3f}")
            _unit_square(ax)
            ax.set(ylabel="observed fraction" if k == 0 else None, xlabel=f"calibrated P({name})")
            fs.style_axes(ax)
            if not yk.size:
                _empty(ax)
        _row_label(row[-1], _who(m, sd))
    for ax, name in zip(grid[0], CLASSES):
        ax.set_title(f"P({name})")
    _tag(fig, "Per-class calibration", split=split, fold=fold)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _unit_calibration(T: Dict[str, Any], m: str, sd: str, fold: Any) -> Dict[str, Any]:
    """The val ``calibration.json`` of the neural unit behind ``m`` in ``fold`` (a ``<kind>_covoff`` or
    ``<kind>_stage`` pass shares its unit's), with ``temperature`` (1/a for Platt); {} for a baseline."""
    kind = m.removesuffix("_covoff").removesuffix("_stage")
    rec = T["calibration"].get(f"fold_{fold}/seed_{sd}" + ("" if kind == "model" else f"/{kind}")) or {}
    return rec | ({"temperature": 1.0 / rec["a"]} if "temperature" not in rec and rec.get("a") else {})


def _calibration_folds(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", **_: Any) -> Any:
    """K4: rows = models; the calibration slope and intercept of each fold's GUID score (● calibrated, ◦ uncalibrated;
    pooled last) and the temperature each fold's unit fitted on val (``calibration.json``; baselines have none);
    reference lines at slope 1, intercept 0 and T = 1 (symlog / log x)."""
    fs, key = _seam(), _Once()
    d = _sel(T["metrics"], level="guid", split=split, metric_type="threshold_free")
    d = d[d["metric"].isin(["calib_slope", "calib_intercept"]) & (d["subgroup"].isna() | (d["subgroup"] == "calibration"))
          ] if len(d) else d
    models = _models(d)
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=1.5, width=9.0, sharex=False)
    for row, (m, sd) in zip(grid, models):
        x = _sel(d, model_id=m, seed=sd)
        folds = sorted((f for f in x["fold"].unique() if f != "pooled"), key=int) + ["pooled"]
        ys = {f: -i for i, f in enumerate(folds)}
        ticks = [f if f == "pooled" else f"fold {f}" for f in folds]
        for ax, met, ref in ((row[0], "calib_slope", 1.0), (row[1], "calib_intercept", 0.0)):
            for cal, face in ((True, fs.BLUE), (False, "white")):
                v = _sel(x, metric=met)
                v = v[v["subgroup"].isna() == cal].drop_duplicates("fold")
                if len(v):
                    ax.plot(v["value"].astype(float), v["fold"].map(ys), "o", ms=5, mfc=face,
                            mec=fs.EDGE if cal else fs.BLUE, mew=fs.MARKER_EDGE if cal else 1.0, ls="none",
                            label="calibrated" if cal else "uncalibrated", zorder=3)
            ax.axvline(ref, ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
            if _sel(x, metric=met)["value"].astype(float).abs().max() > 10:
                ax.set_xscale("symlog", linthresh=1.0)  # separated folds fit slopes and intercepts in the hundreds
            ax.set_yticks(list(ys.values()), ticks if met == "calib_slope" else [""] * len(ys))
            ax.set_ylim(min(ys.values()) - 0.6, 0.6)
            key(ax)
            fs.style_axes(ax, grid="x")
        temps = [(f, _unit_calibration(T, m, sd, f).get("temperature")) for f in folds[:-1]]
        temps = [(f, t) for f, t in temps if t is not None and np.isfinite(t) and t > 0]
        _row_label(row[2], _who(m, sd))
        if not temps:
            _empty(row[2])
            row[2].set(xticks=[], yticks=[])
            continue
        row[2].plot([t for _, t in temps], [ys[f] for f, _ in temps], "o", ms=5, color=fs.BLUE, mec=figstyle.EDGE, mew=0.6)
        row[2].axvline(1.0, ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        row[2].set_xscale("log")
        row[2].set_yticks(list(ys.values()), [""] * len(ys))
        row[2].set(ylim=row[0].get_ylim())
        fs.style_axes(row[2], grid="x")
    for ax, title, xl in zip(grid[0], ("Calibration slope", "Calibration intercept", "Temperature (val fit)"),
                             ("slope", "intercept", "temperature T")):
        ax.set_title(title)
    for ax, xl in zip(grid[-1], ("slope", "intercept", "temperature T")):
        ax.set_xlabel(xl)
    _tag(fig, "Calibration per fold", split=split)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _prevalence_shift(T: Dict[str, Any], c: Dict[str, Any], *, fold: Optional[str] = None, **_: Any) -> Any:
    """K5 (test): rows = models; (1) the reliability of the calibrated GUID probability as fitted (to each fold's val
    prevalence) and after the prior-shift correction to each fold's test prevalence (``metrics.prior_shifted``); (2) PPV
    and (3) NPV vs prevalence of every GUID-level policy from its test sensitivity and specificity (Bayes,
    ``adjust_ppv_npv``): ● the observed test value at the test prevalence, ◦ at the val prevalence, dashed
    ``eval.reference_prevalence``."""
    from matplotlib.lines import Line2D

    fs, one, pi_ref = _seam(), _one(fold), c["eval"].get("reference_prevalence")
    key, ppv_key = _Once(), _Once()
    models = _models(_sel(T["guids"], split="test"))
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=2.3, width=9.6, sharex="col")
    xs = np.linspace(0.005, 0.995, 199)
    for row, (m, sd) in zip(grid, models):
        g = _sel(T["guids"], model_id=m, seed=sd).dropna(subset=["score_final_cal"])
        test, val = g[g["split"] == "test"], g[g["split"] == "val"]
        if len(test) and len(val):
            t = test.assign(shifted=prior_shifted(test, val), unit=test["guid"])
            t = t[t["fold"].astype(str) == one] if fold is not None else pool_rows(t, c["data"]["shared_test_policy"], "test")
            for col, color, what in (("score_final_cal", fs.BLUE, "as fitted"),
                                     ("shifted", fs.ROSE, "prior-shift corrected")):
                _rel_plot(row[0], _reliability(t["y"], expit(t[col].to_numpy(np.float64))), color, what)
        _rel_frame(row[0], "", "observed fraction adverse", key)
        if not len(test) or not len(val):
            _empty(row[0])
        met = _sel(T["metrics"], model_id=m, seed=sd, level="guid", split="test", fold=one)
        met = met[met["subgroup"].isna() & met["policy_id"].notna() & (met["policy_id"] != "oracle")] if len(met) else met
        prev = _calib_rows(T, m, sd, "test", fold, "prevalence_shift", "prevalence")
        pi_val = prev.drop_duplicates("metric").set_index("metric")["value"].get("pi_val") if len(prev) else None
        for ax, what, j in ((row[1], "ppv", 0), (row[2], "npv", 1)):
            for pid in _pids(c, met["policy_id"].unique() if len(met) else []):
                v = _sel(met, policy_id=pid).drop_duplicates("metric").set_index("metric")
                if not {"sens", "spec", what} <= set(v.index):
                    continue
                color, sens, spec = _policy_color(c, pid), v.at["sens", "value"], v.at["spec", "value"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ax.plot(xs, adjust_ppv_npv(sens, spec, xs)[j], color=color, lw=fs.LINE_REGULAR * 1.2, label=pid)
                    pt = v.at["sens", "n_pos"] / (v.at["sens", "n_pos"] + v.at["sens", "n_neg"])
                    ax.plot(pt, v.at[what, "value"], "o", ms=5, color=color, mec=figstyle.EDGE, mew=0.6, zorder=3)
                    if pi_val is not None and np.isfinite(pi_val):
                        ax.plot(pi_val, adjust_ppv_npv(sens, spec, pi_val)[j], "o", ms=5, mfc="white", mec=color,
                                zorder=3)
            if not ax.lines:
                _empty(ax)
                continue
            if pi_ref is not None:
                ax.axvline(pi_ref, ls=(0, (4, 2)), color=fs.MUTED, lw=fs.LINE_REGULAR, label=f"π_ref {pi_ref:g}")
            ax.set(xscale="log", ylabel=what.upper())
            _unit_y(ax)
            if what == "ppv":
                ppv_key(ax)
            fs.style_axes(ax)
        _row_label(row[2], _who(m, sd))
    fs.add_key(fig, [Line2D([], [], marker="o", ms=5, ls="none", mfc="white", mec=fs.MUTED)],
               ["hollow: validation π"])
    for ax, title in zip(grid[0], ("Reliability under prior shift", "PPV vs prevalence", "NPV vs prevalence")):
        ax.set_title(title)
    for ax, xl in zip(grid[-1], ("predicted probability", "prevalence π (log)", "prevalence π (log)")):
        ax.set_xlabel(xl)
    _tag(fig, "Prevalence shift", fold=fold)
    return fig


def _decision_curve(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                    **_: Any) -> Any:
    """K6: per model, the net benefit of treating when the calibrated GUID probability exceeds p_t, vs p_t: pooled thick
    with its patient-cluster bootstrap band, per fold thin; treat-all dashed, treat-none the zero line. The keys are
    drawn once."""
    fs, one, key = _seam(), _one(fold), _Once()
    d = _sel(T["decision_curve"], split=split)
    models = _models(d)
    if not models:
        return _all_empty()
    fig, axes = _panels(len(models), 3, 2.4)
    for ax, (m, sd) in zip(axes, models):
        x = _sel(d, model_id=m, seed=sd)
        ax.set_title(_who(m, sd))
        if fold is None:
            for i, (_, f) in enumerate(x[x["fold"] != "pooled"].groupby("fold")):
                ax.plot(f["pt"], f["net_benefit"], color=fs.BLUE, lw=fs.LINE_THIN, alpha=0.25,
                        label="per fold" if i == 0 else None)
        p = x[x["fold"] == one].sort_values("pt")
        if not len(p):
            _empty(ax)
            continue
        ax.fill_between(p["pt"], p["nb_lo"].astype(float), p["nb_hi"].astype(float), color=fs.BLUE, alpha=0.16, lw=0)
        ax.plot(p["pt"], p["net_benefit"], color=fs.BLUE, lw=fs.LINE_EMPHASIS * 2, label="model", zorder=3)
        ax.plot(p["pt"], p["treat_all"], color=fs.ORANGE, lw=fs.LINE_REGULAR, label="treat all")
        ax.axhline(0.0, color=fs.INK, lw=fs.LINE_HAIRLINE, label="treat none")
        top = float(np.nanmax([p["net_benefit"].max(), p["treat_all"].max(), 0.01]))
        ax.set(xlim=(p["pt"].min(), p["pt"].max()), ylim=(-0.25 * top, 1.15 * top), xlabel="threshold probability $p_t$",
               ylabel="net benefit")
        key(ax)
        fs.style_axes(ax)
    _tag(fig, "Decision curves", split=split, fold=fold)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _brier_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                   fold: Optional[str] = None) -> Any:
    """K7: per model, the snapshot Brier score vs time (bins; the instantaneous population) of the calibrated adverse
    probability and, on a 3-class task, of each calibrated class probability (one-vs-rest, class colours) and their
    macro mean; pooled thick with its 95% band, folds thin; the primary model's n strip under each column."""
    fs, one, key = _seam(), _one(fold), _Once()
    d = _sel(_tr(T, level="online", axis=axis, split=split), point="bin", metric_type="threshold_free",
             denominator="bin_present")
    d = d[d["subgroup"].isna() & d["metric"].astype(str).str.startswith("brier")]
    models = _models(d)
    if not models:
        return _all_empty()
    fig, axes, strips = _model_panels(models)
    _xlim(fig.axes, d["t"], axis)
    colors = fs.group_colors(CLASSES)
    lines = [("brier", fs.BLUE, "o", "adverse probability", "-"),
             *((f"brier_c{k}", colors[n], "s", f"P({n})", "-") for k, n in enumerate(CLASSES)),
             ("brier_macro", fs.INK, "D", "macro", "-")]
    for a, (m, sd) in zip(axes, models):
        x = _sel(d, model_id=m, seed=sd)
        three = bool(len(_sel(x, metric="brier_c0")))  # 1 - P(healthy) is the adverse probability: one line, not two
        for met, color, marker, label, ls in lines:
            f = _sel(x, metric=met)
            if len(f) and not (three and met == "brier"):
                _series(a, f, c, color=color, marker=marker, label=label, fold=fold, ls=ls)
        if a.lines:
            a.set_ylim(bottom=0)
            fs.style_axes(a)
        else:
            _empty(a)
        a.set(ylabel="Brier score", title=_who(m, sd))
        _time_lines(a, axis)
        key(a)
    sm = _strip_model(models)
    for j, s in enumerate(strips):
        _n_strip(s, _sel(d, model_id=sm[0], seed=sm[1], metric="brier", fold=one), c, legend=j == 0)
        _time_lines(s, axis)
        _x_time(s, axis)
    _tag(fig, "Brier score over time", split=split, fold=fold)
    fs.caveat_note(fig, text=f"Hollow, black ring: no estimate (fewer than {c['eval']['min_bin_class_n']} GUIDs of a class in the bin; "
                             "the line is joined over it). "
                             f"n strip: {_who(*sm)}. {_fold_phrase(T, c, split)}".rstrip())
    return fig


def _forest(ax: Any, v: pd.DataFrame, ys: Dict[str, float], color: str, marker: str, dy: float,
            label: Optional[str]) -> None:
    """Per-fold points ``value`` at ``ys[fold] + dy`` with their 95% CI bars (NaN CIs draw none)."""
    fs = _seam()
    v = v[v["fold"].isin(list(ys))]
    if not len(v):
        return
    x = v["value"].astype(float).to_numpy()
    err = np.clip([x - v["ci_lo"].astype(float).to_numpy(), v["ci_hi"].astype(float).to_numpy() - x], 0, None)
    ax.errorbar(x, v["fold"].map(ys).to_numpy(float) + dy, xerr=err, fmt=marker, ms=4.5, color=color, mec=figstyle.EDGE,
                mew=0.5, elinewidth=fs.LINE_REGULAR, label=label, zorder=3)


def _fold_forest(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", **_: Any) -> Any:
    """H1: rows = models; per fold (N and prevalence in the label) and pooled: (1) AUROC ● and pAUC@alpha ◆ with 95%
    CIs, the fold mean dashed, the I² of the per-fold AUROC in the panel corner; (2) sensitivity and (3) FPR overshoot
    (test FPR - alpha) of every policy, with 95% CIs."""
    fs, alpha, key, pol_key = _seam(), primary_alpha(c["eval"]), _Once(), _Once()
    d = _sel(T["metrics"], level="guid", split=split)
    models = _models(d)
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=1.9, width=11.0, sharex=False)
    for row, (m, sd) in zip(grid, models):
        x = _sel(d, model_id=m, seed=sd)
        base, het = x[x["subgroup"].isna()], _sel(x, subgroup="fold_heterogeneity", fold="pooled")
        tf = base[base["metric_type"] == "threshold_free"]
        au = _sel(tf, metric="auroc").drop_duplicates("fold")
        folds = sorted((f for f in au["fold"] if f != "pooled"), key=int) + ["pooled"]
        ys = {f: -float(i) for i, f in enumerate(folds)}
        prev = _sel(tf, metric="prevalence").drop_duplicates("fold").set_index("fold")["value"]
        n = au.set_index("fold")[["n_pos", "n_neg"]].astype(float).sum(1)  # a fold's N is its weighted GUID count
        ticks = [f"{f if f == 'pooled' else f'fold {f}'} (N {format(n.get(f, 0), _count_spec(n.get(f, 0)))}, "
                 f"π {prev.get(f, np.nan):.2f})" for f in folds]
        for ax in row:
            ax.axhspan(-len(folds) + 0.5, -len(folds) + 1.5, color=fs.STRIP, lw=0, zorder=0)  # the pooled row
        _forest(row[0], au, ys, fs.BLUE, "o", 0.0, "AUROC")
        _forest(row[0], _sel(tf, metric=f"pauc@{alpha:g}").drop_duplicates("fold"), ys, fs.VIOLET, "D", -0.2,
                f"pAUC@{alpha:g}")
        per = au[au["fold"] != "pooled"]["value"].astype(float)
        i2 = het.drop_duplicates("metric").set_index("metric")["value"].get("i2") if len(het) else None
        if len(per):
            row[0].axvline(per.mean(), ls=(0, (4, 2)), color=fs.BLUE, lw=fs.LINE_HAIRLINE, label="fold mean AUROC")
        _stat(row[0], "I² " + (f"{i2:.2f}" if i2 is not None and np.isfinite(i2) else "n/a"), "lower left")
        pol = base[base["policy_id"].notna() & (base["policy_id"] != "oracle")]
        pids = _pids(c, pol["policy_id"].unique())
        for ax, met in ((row[1], "sens"), (row[2], "fpr_overshoot")):
            for i, pid in enumerate(pids):
                v = _sel(pol, policy_id=pid, metric=met).drop_duplicates("fold")
                if len(v):
                    _forest(ax, v, ys, _policy_color(c, pid), "o", -0.6 * (i + 0.5) / max(len(pids), 1) + 0.3, pid)
            if met == "fpr_overshoot" and ax.lines:
                ax.axvline(0.0, ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        for ax in row:
            if not ax.lines and not ax.collections:
                _empty(ax)
                continue
            ax.set_yticks(list(ys.values()), ticks if ax is row[0] else [""] * len(ys))
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(min(ys.values()) - 0.55, 0.55)
            fs.style_axes(ax, grid="x")
        key(row[0])
        pol_key(row[1])
        _row_label(row[2], _who(m, sd))
    for ax, title, xl in zip(grid[0], ("AUROC and pAUC", "Sensitivity", "FPR overshoot"),
                             ("AUROC / pAUC", "sensitivity", "test FPR − α")):
        ax.set_title(title)
    for ax, xl in zip(grid[-1], ("AUROC / pAUC", "sensitivity", "test FPR − α")):
        ax.set_xlabel(xl)
    _tag(fig, "Fold heterogeneity", split=split)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _seed_spread(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", **_: Any) -> Any:
    """H2 (more than one seed): per model, the GUID AUROC of every seed per fold (small dots) and pooled (● with CI),
    the seed ensemble ``ens`` in rose (small dots ``fold``, large ``pooled`` with its 95% CI)."""
    from matplotlib.lines import Line2D

    fs = _seam()
    d = _sel(T["metrics"], level="guid", split=split, metric="auroc", metric_type="threshold_free")
    d = d[d["subgroup"].isna()] if len(d) else d
    names = list(dict.fromkeys(d["model_id"])) if len(d) else []
    if not names:
        return _all_empty()
    fig, axes = _panels(len(names), 3, 2.3)
    for ax, m in zip(axes, names):
        x = _sel(d, model_id=m)
        seeds = sorted(x["seed"].astype(str).unique(), key=lambda s: (s == "ens", s))
        for i, s in enumerate(seeds):
            v, color = _sel(x, seed=s), fs.ROSE if s == "ens" else fs.BLUE
            per, p = v[v["fold"] != "pooled"], v[v["fold"] == "pooled"]
            ax.plot(np.full(len(per), i) + np.linspace(-0.12, 0.12, len(per)), per["value"].astype(float), "o", ms=3,
                    alpha=0.5, mew=0, color=color)
            if len(p):
                val, lo, hi = (float(p[k].iloc[0]) for k in ("value", "ci_lo", "ci_hi"))
                ax.errorbar(i, val, yerr=np.clip([[val - lo], [hi - val]], 0, None), fmt="o", ms=6, color=color,
                            mec=figstyle.EDGE, mew=0.7, elinewidth=fs.LINE_REGULAR * 1.3, zorder=3)
        ax.set_xticks(range(len(seeds)), seeds, fontsize=fs.FONT_SMALL)
        ax.set(xlabel="seed", ylabel="AUROC", title=m, xlim=(-0.6, len(seeds) - 0.4))
        fs.style_axes(ax, grid="y")
    fs.add_key(fig, [Line2D([], [], marker="o", ms=3, ls="none", color=fs.BLUE, alpha=0.5),
                     Line2D([], [], marker="o", ms=6, ls="-", color=fs.BLUE, mec=figstyle.EDGE),
                     Line2D([], [], marker="o", ms=6, ls="-", color=fs.ROSE, mec=figstyle.EDGE)],
               ["fold", "pooled, 95% CI", "seed ensemble"])
    _tag(fig, "Seed spread", split=split)
    fs.caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _roc_stage(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
               **_: Any) -> Any:
    """R10: one panel per checkpoint and end, the snapshot ROC restricted to first-stage (solid) and second-stage
    (dashed) snapshots, every model overlaid; pooled thick, per fold thin; each model's first | second AUC in its
    colour in the panel corner."""
    from matplotlib.lines import Line2D

    fs, one = _seam(), _one(fold)
    r = _sel(T["roc"], level="online", split=split)
    r = r[r["variant"].astype(str).str.startswith(("snapshot_first@", "snapshot_second@"))] if len(r) else r
    ats = [f"{h:g}" for h in c["eval"]["checkpoints_h"]] + ["end"]
    models, palette = _models(r), fs.LINE_PALETTE
    fig, axes = _panels(len(ats), 4, 2.3)
    for ax, at in zip(axes, ats):
        stats = []
        for i, (m, sd) in enumerate(models):
            color, aucs = palette[i % len(palette)], []
            for stage, shade in (("first", color), ("second", fs.tint(color, 0.5))):
                g = _sel(r, model_id=m, seed=sd, variant=f"snapshot_{stage}@{at}")
                if fold is None:
                    for _, h in g[g["fold"] != "pooled"].groupby("fold"):
                        ax.plot(h["fpr"], h["tpr"], color=shade, lw=fs.LINE_THIN * 0.8, alpha=0.2)
                p = g[g["fold"] == one]
                aucs.append(np.trapezoid(p["tpr"].to_numpy(float), p["fpr"].to_numpy(float)) if len(p) else np.nan)
                if len(p):
                    ax.plot(p["fpr"], p["tpr"], color=shade, lw=fs.LINE_EMPHASIS * 1.8,
                            label=_who(m, sd) if stage == "first" else None, zorder=3)
            if np.isfinite(aucs).any():
                stats.append((color, " | ".join("–" if a != a else f"{a:.2f}" for a in aucs)))
        ax.set_title("All segments" if at == "end" else f"{at} h before delivery")
        if not ax.lines:
            _empty(ax)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            continue
        _key(ax)
        _unit_square(ax)
        _stat_list(ax, stats)
        ax.set(xlabel="FPR", ylabel="sensitivity")
        fs.style_axes(ax)
    fs.add_key(fig, [Line2D([], [], color=fs.MUTED, lw=2), Line2D([], [], color=fs.tint(fs.MUTED, 0.5), lw=2)],
               ["first stage (full colour)", "second stage (light shade)"])
    _tag(fig, "Stage-specific snapshot ROC", split=split, fold=fold)
    fs.caveat_note(fig, text=f"Numbers: AUC first | second stage of each model, in its colour. {_fold_phrase(T, c, split)}"
                   .rstrip())
    return fig


def _score_distributions(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                         **_: Any) -> Any:
    """R12: per model, violins (inner box) of the calibrated adverse probability per clinical class: the GUID final
    score, then the snapshot at every checkpoint (``snapshots.parquet``; pooled test, each GUID once); a model with
    3-class probabilities adds a row, one panel per calibrated class probability, one violin per true class. Each row
    is named in its facet strip."""
    fs = _seam()
    ckpts = [float(h) for h in c["eval"]["checkpoints_h"]]
    g = _sel(T["guids"], split=split)
    models = _models(g)
    if not models:
        return _all_empty()
    snap = _sel(T["snapshots"], split=split, axis="to_delivery", point="checkpoint")
    snap = snap[snap["fold"].astype(str) == str(fold)] if fold is not None and len(snap) else snap
    three = [(m, sd) for m, sd in models
             if set(P3_CAL) <= set(g.columns) and len(_guid_rows(T, c, m, sd, split, fold, P3_CAL))]
    specs = [(m, sd, p3) for m, sd in models for p3 in (False, True) if not p3 or (m, sd) in three]
    fig, axes = _figure(len(specs), 1 + len(ckpts), 1.95, width=fs.WINDOWS_FIGURE_WIDTH)
    for row, (m, sd, p3) in zip(axes, specs):
        x = _guid_rows(T, c, m, sd, split, fold, P3_CAL if p3 else ("score_final_cal",))
        order = _order(x["clinical_class"].astype(str).unique(), "clinical_class") if len(x) else list(CLASSES)
        colors = fs.group_colors(order)

        def violins(ax: Any, frame: pd.DataFrame, values: Any, title: str, ylabel: str = "") -> None:
            cls = frame["clinical_class"].astype(str).to_numpy() if len(frame) else np.zeros(0, str)
            fs.violin_panel(ax, {k: np.asarray(values, np.float64)[cls == k] for k in order}, title=title, ylabel=ylabel,
                            colors=colors)
            ax.set_ylim(-0.04, 1.04)
            fs.style_axes(ax, grid="y")

        if p3:
            for k, (ax, name) in enumerate(zip(row, CLASSES)):
                violins(ax, x, x[P3_CAL[k]], f"P({name})", "probability" if k == 0 else "")
            for ax in row[3:]:
                ax.set_visible(False)
            _row_label(row[2], f"{_who(m, sd)} · 3-class")
            continue
        violins(row[0], x, expit(x["score_final_cal"].to_numpy(np.float64)), "Final score", "P(adverse)")
        s = _sel(snap, model_id=m, seed=sd)
        for ax, h in zip(row[1:], ckpts):
            v = s[np.isclose(s["t"].astype(float), h)] if len(s) else s
            violins(ax, v, expit(v["score"].to_numpy(np.float64)) if len(v) else np.zeros(0), f"{h:g} h before delivery")
        _row_label(row[-1], _who(m, sd))
    _tag(fig, "Calibrated score by class", split=split, fold=fold)
    return fig


#: R13 draws at most this many windows (three violins each) on its fixed-width page; more bins are merged.
R13_MAX_WINDOWS = 12


def _merge_windows(d: pd.DataFrame, width: float, axis: str, most: int = R13_MAX_WINDOWS) -> Tuple[pd.DataFrame, float]:
    """``(snapshots, window width)`` with at most ``most`` windows: when the bin grid has more (a 12-hour window of
    half-hour bins has 24), ``k`` neighbouring bins merge into one window of ``k * width`` and each GUID keeps its
    latest snapshot in it (the snapshot rule, so a GUID counts once per window); ``t`` becomes the window centre."""
    if d.empty:
        return d, width
    t = d["t"].astype(float)
    k = int(np.ceil(t.nunique() / most))
    if k <= 1:
        return d, width
    i = np.rint((t - t.min()) / width).astype(int) // k
    latest = t.rank(method="first", ascending=axis != "to_delivery")  # to_delivery: fewer hours left is later
    d = d.assign(t=t.min() + (i * k + (k - 1) / 2.0) * width, _w=i, _r=latest)
    keep = d.sort_values("_r").drop_duplicates(["model_id", "seed", "fold", "guid", "_w"], keep="last")
    return keep.drop(columns=["_w", "_r"]), k * width


def _score_windows(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                   fold: Optional[str] = None) -> Any:
    """R13: ``windowed_comparison_figure`` of the calibrated snapshot probability by clinical class per window of
    ``axis`` (``eval.bin_h``; 1 segment on position), pooled test, one readout per model that is not a baseline: the
    violins, the Kruskal-Wallis + Holm strip and Cliff's delta of the surviving pairs (``stats.windowed_group_
    comparisons``; a cell below ``MIN_GROUP_SIZE`` GUIDs is drawn but left out of the test)."""
    from teb_vae.lag_attn.eval import stats

    fs = _seam()
    d = _sel(T["snapshots"], split=split, axis=axis, point="bin")
    d = d[d["fold"].astype(str) == str(fold)] if fold is not None and len(d) else d
    models = [x for x in _models(d) if x[0] not in BASELINES] or _models(d)  # ponytail: baselines only when alone
    if not models:
        return _all_empty()
    d, width = _merge_windows(d, 1.0 if axis == "position" else c["eval"]["bin_h"], axis)
    order = _order(d["clinical_class"].astype(str).unique(), "clinical_class")
    readouts = []
    for m, sd in models:
        x = _sel(d, model_id=m, seed=sd)
        x = x.assign(p=expit(x["score"].to_numpy(np.float64)), cls=x["clinical_class"].astype(str))
        ts = sorted(x["t"].astype(float).unique())
        cells = [{k: v["p"].to_numpy() for k in order for v in [w[w["cls"] == k]] if len(v)}
                 for t in ts for w in [x[x["t"].astype(float) == t]]]
        usable = {t: {k: v for k, v in cell.items() if v.size >= stats.MIN_GROUP_SIZE} for t, cell in zip(ts, cells)}
        rec = stats.windowed_group_comparisons(usable, meta_by_window={t: {"bin_center_h": t} for t in ts})
        readouts.append((_who(m, sd), cells, rec))
    fig = fs.windowed_comparison_figure(readouts, groups=order, bin_width=width,
                                        min_body_size=stats.MIN_GROUP_SIZE, xlabel=AXIS_LABEL.get(axis, axis),
                                        ylabel="P(adverse), calibrated", delivery_orientation=axis == "to_delivery")
    for ax in fig.axes:  # the shared panels key inside the axes (loc="best"); restyled like every in-panel legend here
        if ax.get_legend():
            _legend(ax)
    _tag(fig, "Score by class per window", split=split, fold=fold)
    return fig


def _num3(v: Any) -> str:
    """3 significant digits (a fitted temperature spans 1e-6 to 1e2); '-' when absent."""
    return "-" if v is None or not np.isfinite(v) else f"{v:.3g}"


def _join(s: pd.Series) -> str:
    return ", ".join(f"{k}: {_cell(v)}" for k, v in s.items()) or "-"


def _heterogeneity_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§10 H1 and H4. H1, per model (test, GUID level): the per-fold AUROC with its mean ± SD, range and I² (block H),
    the primary policy's per-fold sensitivity and FPR overshoot, and the per-fold prevalence. H4: validation vs test for
    the headline AUROC, pAUC and primary-policy rates (``results.headline``; optimism = val - test, on the fold mean for
    AUROC/pAUC and pooled otherwise)."""
    primary, alpha = c["eval"]["primary_policy"], primary_alpha(c["eval"])
    d = _sel(T["metrics"], level="guid", split="test")
    lines = []
    for (m, sd), x in (d.groupby(["model_id", "seed"], sort=False) if len(d) else ()):
        base = x[x["subgroup"].isna() & (x["fold"] != "pooled")]

        def per(metric: str, pid: Optional[str] = None, base: pd.DataFrame = base) -> pd.Series:
            v = _sel(base, metric=metric)
            v = v[v["policy_id"].isna()] if pid is None else _sel(v, policy_id=pid)
            return v.drop_duplicates("fold").set_index("fold")["value"].astype(float)

        au = per("auroc")
        het = _sel(x, subgroup="fold_heterogeneity", fold="pooled").drop_duplicates("metric").set_index("metric")["value"]
        i2 = het.get("i2")
        i2s = (f"I² {i2:.2f} (Cochran's Q {het.get('cochran_q'):.2f}, p {het.get('cochran_q_p'):.3f})"
               if i2 is not None and np.isfinite(i2) else "I² n/a (a fold's AUROC CI has zero width, or < 2 folds)")
        lines.append(f"- **{m}** (seed {sd}): AUROC per fold {_join(au)}; mean ± SD {_cell(au.mean())} ± {_cell(au.std())}, "
                     f"range {_cell(au.min())}-{_cell(au.max())}; {i2s}. `{primary}` sensitivity per fold "
                     f"{_join(per('sens', primary))}; FPR overshoot {_join(per('fpr_overshoot', primary))}; prevalence "
                     f"{_join(per('prevalence'))}.")
    head = pd.DataFrame(list(((T["summary"].get("results") or {}).get("headline") or {}).values()))
    h4 = []
    if len(head):
        head = head[(head["level"] == "guid") & head["policy_id"].isin([primary, "threshold_free"])
                    & head["metric"].isin(["auroc", f"pauc@{alpha:g}", "sens", "spec", "fpr"])]
        for r in head.itertuples(index=False):
            fm = r.primary == "fold_mean"
            pair = (r.val_fold_mean, r.fold_mean) if fm else (r.val, r.test)
            val, test = (pd.to_numeric(v, errors="coerce") for v in pair)
            h4.append({"model": r.model_id, "seed": r.seed, "policy": r.policy_id, "metric": r.metric,
                       "estimate": "fold mean" if fm else "pooled", "val (optimistic)": val, "test": test,
                       "optimism (val - test)": val - test})
    return ["### Fold heterogeneity (H1)", "",
            "Test, GUID level, per fold (the forest: `heterogeneity/fold_forest`). I²: Cochran's Q of the per-fold AUROC "
            "with inverse-variance weights (SE from the bootstrap CI). " + _fold_def(T, c, short=True), "",
            *(lines or ["_no per-fold rows_"]), "",
            "### Validation vs test (H4)", "",
            "Validation values are optimistic (used for selection); optimism = val - test.", "", _md(pd.DataFrame(h4)), ""]


_BUILDERS.update({
    CALIBRATION_GUID: _calibration_guid, CALIBRATION_PER_CLASS: _calibration_per_class,
    CALIBRATION_FOLDS: _calibration_folds, PREVALENCE_SHIFT: _prevalence_shift, DECISION_CURVE: _decision_curve,
    FOLD_FOREST: _fold_forest, SEED_SPREAD: _seed_spread, ROC_STAGE: _roc_stage, SCORE_DISTRIBUTIONS: _score_distributions,
})
AXIS_BUILDERS.update({BRIER_VS_TIME: _brier_vs_time, SCORE_WINDOWS: _score_windows})
EXPECTED_WHEN[CALIBRATION_PER_CLASS] = lambda c: (c.get("labels") or {}).get("task") == "three_class"
EXPECTED_WHEN[SEED_SPREAD] = lambda c: len((c.get("run") or {}).get("seeds") or []) > 1
CORE_EXTRA.append(CALIBRATION_GUID)
EXTRA_TABLES.extend(["decision_curve", "snapshots"])

# ---- block X (confusion and 3-class: X1-X9) ----
from typing import Tuple  # noqa: E402

from teb_vae.classifier.metrics import (  # noqa: E402
    ARGMAX_TR, CLASSES, CONFUSION_NAMES, ROC_GRID, aux_collapse, confusion_stats, fold_mean_rownorm, row_normalised,
    tpr_at,
)

CONFUSION_BINARY = "confusion/confusion_binary"
CONFUSION_3CLASS = "confusion/confusion_3class"
ROC_OVR = "multiclass/roc_ovr"
PR_OVR = "multiclass/pr_ovr"
CONFUSION_EVOLUTION = "multiclass/confusion_evolution_{axis}"
PER_CLASS_VS_TIME = "multiclass/per_class_vs_time_{axis}"
PER_CLASS_AUROC_VS_TIME = "multiclass/per_class_auroc_vs_time_{axis}"
F1_VS_TIME = "multiclass/f1_vs_time_{axis}"
COLLAPSE_VS_BINARY = "multiclass/collapse_vs_binary"
SHORT_CLASS = ("healthy", "acid.", "HIE")
#: The M engine's per-policy rate rows :func:`_rates` reads, as X5 reads them per class.
OVR_RATES = ("tp", "fp", "sens", "spec", "fpr", "underpowered")


def _is_three_class(c: Dict[str, Any]) -> bool:
    return (c.get("labels") or {}).get("task") == "three_class"


def _is_multi_task(c: Dict[str, Any]) -> bool:
    """A binary task with the aux 3-class head (X9)."""
    lab = c.get("labels") or {}
    return lab.get("task") != "three_class" and (lab.get("aux_3class_weight") or 0) > 0


def _texts(field: np.ndarray, spec: str) -> np.ndarray:
    """Cell labels of ``field`` in ``spec`` ('-' for NaN), same shape."""
    return np.array(["-" if v != v else format(v, spec) for v in np.ravel(field)], dtype=object).reshape(np.shape(field))


def _heat(fig: Any, ax: Any, field: np.ndarray, text: np.ndarray, *, title: str, xticks: Any, yticks: Any,
          vmax: float = 1.0, **kw: Any) -> None:
    """``field`` on ``[0, vmax]`` in the sequential blue map through the seam's empty-safe heatmap, ``text`` in each
    cell (white on the dark half), white gaps between the cells."""
    fs = _seam()
    if fs.heatmap_with_colorbar(fig, ax, field, title=title, symmetric=False, vlimits=(0, vmax), cmap=fs.SEQUENTIAL_CMAP,
                                **kw) is None:
        return
    for (i, j), s in np.ndenumerate(text):
        v = field[i, j]
        ax.text(j, i, s, ha="center", va="center", fontsize=fs.FONT_SMALL,
                color="white" if v == v and v > 0.55 * vmax else fs.INK)
    ax.set_xticks(range(len(xticks)), xticks, fontsize=fs.FONT_SMALL)
    ax.set_yticks(range(len(yticks)), yticks, fontsize=fs.FONT_SMALL)
    ax.set_xticks(np.arange(-0.5, len(xticks)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(yticks)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def _confusion_binary(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                      **_: Any) -> Any:
    """X1: per model, every threshold policy's confusion (rows: GUID-level policies, then segment-level ones) as TP |
    FN | FP | TN counts on the policy's own basis population (pooled: summed over folds, §11.7), coloured and annotated
    with the rate within the true class (sensitivity, miss rate, FPR, specificity). A 3-class run's binary policies
    threshold the collapsed adverse score."""
    d = _sel(T["metrics"], split=split, fold=_one(fold))
    if len(d):
        d = d[d["subgroup"].isna() & d["level"].isin(["guid", "segment"]) & d["metric"].isin(["tp", "fn", "fp", "tn"])
              & d["policy_id"].notna() & ~d["policy_id"].isin(["oracle", "argmax"])]
    models = _models(d)
    if not models:
        return _all_empty()
    ncol = min(2, len(models))
    n_pol = max(1, len(_sel(d, model_id=models[0][0], seed=models[0][1])[["level", "policy_id"]].drop_duplicates()))
    fig, axes = _figure(-(-len(models) // ncol), ncol, 0.32 * n_pol + 1.0, width=4.1 * ncol + 0.4)
    for ax, (m, sd) in zip(axes.flat, models):
        x = _sel(d, model_id=m, seed=sd).pivot_table(index=["level", "policy_id"], columns="metric", values="value")
        idx = [(lvl, p["id"]) for lvl in ("guid", "segment") for p in c["eval"]["thresholds"] if (lvl, p["id"]) in x.index]
        cnt = (x.reindex(index=pd.MultiIndex.from_tuples(idx), columns=["tp", "fn", "fp", "tn"]).to_numpy(np.float64)
               if idx else np.zeros((0, 4)))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = cnt / np.repeat(np.c_[cnt[:, :2].sum(1), cnt[:, 2:].sum(1)], 2, axis=1)
        text = np.array([f"{a} · {b}" for a, b in zip(_texts(cnt, _count_spec(cnt)).ravel(), _texts(rate, ".2f").ravel())],
                        dtype=object).reshape(rate.shape)
        _heat(fig, ax, rate, text, title=_who(m, sd), xticks=["TP", "FN", "FP", "TN"],
              yticks=[p if lvl == "guid" else f"{p} (segment)" for lvl, p in idx], xlabel="adverse | healthy")
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    _tag(fig, "Confusion per policy", split=split, fold=fold)
    _seam().caveat_note(fig, text=" ".join(filter(None, ("Cell: count · rate within the true class (colour).",
                                                         _fold_phrase(T, c, split) if fold is not None else ""))))
    return fig


def _confusion_3class(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                      **_: Any) -> Any:
    """X2: rows = models; the argmax 3x3 confusion at GUID level: the pooled counts (OOF, summed over folds, §11.7),
    their row-normalised form (recall on the diagonal), and the mean ± SD over folds of the per-fold row-normalised
    matrices (:func:`~teb_vae.classifier.metrics.fold_mean_rownorm`, every fold equal); ``fold``: that fold's two
    matrices."""
    d = _sel(T["metrics"], level="guid", split=split, policy_id="argmax")
    d = d[d["subgroup"].isna() & d["metric"].isin(CONFUSION_NAMES)] if len(d) else d
    models = _models(d)
    if not models:
        return _all_empty(1, 3)
    fig, grid, _ = _facets(len(models), 3, row_h=2.4, width=9.4, sharex=False)
    kw = dict(xticks=SHORT_CLASS, yticks=SHORT_CLASS, xlabel="predicted (argmax)", ylabel="true class")
    for i, (row, (m, sd)) in enumerate(zip(grid, models)):
        mats = {str(f): g.set_index("metric")["value"].reindex(CONFUSION_NAMES).to_numpy(np.float64).reshape(3, 3)
                for f, g in _sel(d, model_id=m, seed=sd).groupby("fold")}
        C = mats.get(_one(fold), np.full((3, 3), np.nan))
        R = row_normalised(C)
        _heat(fig, row[0], C, _texts(C, _count_spec(C)), vmax=max(float(np.nanmax(C, initial=0.0)), 1.0),
              title="Counts" if i == 0 else "", **kw)
        _heat(fig, row[1], R, _texts(R, ".2f"), title="Fraction of true class" if i == 0 else "", **kw)
        per = [v for f, v in mats.items() if f != "pooled"]
        if fold is None and per:
            mean, sdev = fold_mean_rownorm(per)
            _heat(fig, row[2], mean, np.array([f"{a}\n± {b}" for a, b in zip(_texts(mean, ".2f").ravel(), _texts(
                sdev, ".2f").ravel())], dtype=object).reshape(3, 3),
                  title="Fold mean ± SD" if i == 0 else "", **kw)
            _row_label(row[2], _who(m, sd))
        else:
            row[2].set_visible(False)
            _row_label(row[1], _who(m, sd))
    _tag(fig, "Three-class confusion", split=split, fold=fold)
    _seam().caveat_note(fig, text=_fold_phrase(T, c, split))
    return fig


def _argmax_wide(x: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """The argmax snapshot rows of one model (``policy_id = argmax``, time-resolved) as one row per (fold, point, t)
    (``t`` = inf at end) with ``value`` / ``ci_lo`` / ``ci_hi`` columns per metric, and its (rows, 3, 3) confusion
    counts; None without rows."""
    x = x[x["metric"].isin([*CONFUSION_NAMES, *ARGMAX_TR])]
    if not len(x):
        return None
    x = x.assign(fold=x["fold"].astype(str), point=x["point"].astype(str), metric=x["metric"].astype(str),
                 t=x["t"].astype(float).fillna(np.inf))
    w = x.set_index(["fold", "point", "t", "metric"])[["value", "ci_lo", "ci_hi"]].astype(float).unstack("metric")
    return w, w["value"].reindex(columns=list(CONFUSION_NAMES)).to_numpy(np.float64).reshape(-1, 3, 3)


def _argmax_series(wc: Optional[Tuple[pd.DataFrame, np.ndarray]], name: str, mn: int) -> pd.DataFrame:
    """One :func:`_series` frame of an argmax metric on the bins: ``under`` where a class it needs has fewer than
    ``mn`` GUIDs (recall: its own class; the rest: every class), ``raw`` its value from the bin's confusion counts
    (drawn hollow there)."""
    if wc is None:
        return pd.DataFrame(columns=["fold", "t", "value", "ci_lo", "ci_hi", "under", "raw"])
    w, C = wc
    n = C.sum(2)
    under = n[:, int(name[-1])] < mn if name.startswith("recall") else (n < mn).any(1)
    f = pd.DataFrame({k: w[k].get(name) for k in ("value", "ci_lo", "ci_hi")} | {
        "under": under, "raw": confusion_stats(C)[name]}, index=w.index).reset_index()
    return f[f["point"] == "bin"]


def _class_strip(ax: Any, wc: Optional[Tuple[pd.DataFrame, np.ndarray]], c: Dict[str, Any], fold: Optional[str],
                 legend: bool = True) -> None:
    """GUIDs of each class per bin (the snapshot population), the underpowered floor dotted."""
    fs = _seam()
    idx = wc[0].index.to_frame(index=False) if wc else pd.DataFrame(columns=["fold", "point", "t"])
    sel = ((idx["fold"] == _one(fold)) & (idx["point"] == "bin")).to_numpy(bool)
    if not sel.any():
        return _empty(ax)
    t, n = idx["t"].to_numpy(np.float64)[sel], wc[1].sum(2)[sel]
    o, colors = np.argsort(t), fs.group_colors(CLASSES)
    for k, name in enumerate(CLASSES):
        ax.plot(t[o], n[o, k], marker="o", ms=2.5, lw=fs.LINE_REGULAR, color=colors[name], mec=figstyle.EDGE, mew=0.3,
                label=f"{name} (n)")
    ax.axhline(c["eval"]["min_bin_class_n"], ls=":", color=fs.MUTED, lw=fs.LINE_HAIRLINE)
    top = float(np.nanmax(n, initial=1.0))
    ax.set(ylim=(0, 1.2 * top), yticks=[0, round(top)], ylabel="GUIDs\nper bin")
    if legend:
        _key(ax)
    fs.style_axes(ax, grid="x")


def _top_title(fig: Any, text: str) -> None:
    """The figure title (:func:`figstyle.set_title`)."""
    _seam().set_title(fig, text)


def _ovr_rows(x: pd.DataFrame, k: int, names: Tuple[str, ...]) -> pd.DataFrame:
    """Class k's one-vs-rest rows ``ovr_c<k>/<name>`` of ``x`` with the prefix stripped: the M engine's shape."""
    p = f"ovr_c{k}/"
    f = x[x["metric"].isin([p + n for n in names])]
    return f.assign(metric=f["metric"].astype(str).str[len(p):])


def _x_time_rows(T: Dict[str, Any], axis: str, split: str) -> pd.DataFrame:
    """The all-GUID time-resolved bin rows of one axis and split."""
    d = _sel(_tr(T, level="online", axis=axis, split=split), point="bin")
    return d[d["subgroup"].isna()]


def _evolution_points(wc: Optional[Tuple[pd.DataFrame, np.ndarray]], axis: str) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """X3's points of one model: ``(index frame, confusions, picked rows)``, at most 8 non-empty ones: on to_delivery
    the checkpoints (earliest first) and end, elsewhere bins spread evenly over the axis."""
    idx = wc[0].index.to_frame(index=False) if wc else pd.DataFrame(columns=["point", "t"])
    C = wc[1] if wc else np.zeros((0, 3, 3))
    ok, t = C.sum((1, 2)) > 0, idx["t"].to_numpy(np.float64)
    if axis == "to_delivery":
        pick = np.flatnonzero(ok & idx["point"].isin(["checkpoint", "end"]).to_numpy(bool))
        return idx, C, list(pick[np.argsort(np.where(np.isfinite(t[pick]), -t[pick], np.inf), kind="stable")][:8])
    bins = np.flatnonzero(ok & (idx["point"] == "bin").to_numpy(bool))
    bins = bins[np.argsort(t[bins], kind="stable")]
    return idx, C, list(bins[np.unique(np.linspace(0, bins.size - 1, min(8, bins.size)).round().astype(int))]
                        if bins.size else [])


def _confusion_evolution(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                         fold: Optional[str] = None, **_: Any) -> Any:
    """X3: rows = models; the argmax snapshot confusion (each GUID once: its last segment in the window) at up to 8
    points (:func:`_evolution_points`), coloured row-normalised (recall on the diagonal) with the counts annotated."""
    import matplotlib.pyplot as plt

    fs = _seam()
    d = _sel(_tr(T, level="online", axis=axis, split=split), policy_id="argmax", fold=_one(fold))
    d = d[d["subgroup"].isna()]
    models = _models(d)
    if not models:
        return _all_empty()
    pts = [_evolution_points(_argmax_wide(_sel(d, model_id=m, seed=sd)), axis) for m, sd in models]
    n_col = max(1, *(len(p[2]) for p in pts))
    fig, axes = plt.subplots(len(models), n_col, squeeze=False, layout="constrained",
                             figsize=(min(1.35 * n_col + 1.3, fs.WINDOWS_FIGURE_WIDTH), 1.45 * len(models) + 0.9))
    im = None
    for row, (m, sd), (idx, C, pick) in zip(axes, models, pts):
        for j, (ax, i) in enumerate(zip(row, pick)):
            R, t = row_normalised(C[i]), idx["t"].iloc[i]
            im = ax.imshow(R, vmin=0, vmax=1, cmap=fs.SEQUENTIAL_CMAP)
            for (a, b), v in np.ndenumerate(C[i]):
                ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=fs.FONT_SMALL,
                        color="white" if R[a, b] == R[a, b] and R[a, b] > 0.55 else fs.INK)
            ax.set_xticks(range(3), SHORT_CLASS, fontsize=fs.FONT_TINY, rotation=90)
            ax.set_yticks(range(3), SHORT_CLASS if j == 0 else [""] * 3, fontsize=fs.FONT_TINY)
            ax.set_xticks(np.arange(-0.5, 3), minor=True)
            ax.set_yticks(np.arange(-0.5, 3), minor=True)
            ax.grid(which="minor", color="white", lw=1.2)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title("end" if not np.isfinite(t) else f"{t:g}" + ("" if axis == "position" else " h"),
                         fontsize=fs.FONT_SMALL)
        if not pick:
            _empty(row[0])
        row[0].set_ylabel("true class")
        for ax in row[max(len(pick), 1):]:
            ax.set_visible(False)
        _row_label(row[max(len(pick), 1) - 1], _who(m, sd))
    for ax in axes[-1]:
        ax.set_xlabel("predicted")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, aspect=25, label="fraction of true class", pad=0.02)
    _tag(fig, f"Confusion over time · {AXIS_LABEL.get(axis, axis)}", split=split, fold=fold)
    return fig


def _ovr_curve(f: pd.DataFrame, k: int, pr: bool) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """``(x, y, area, prevalence)`` of class k's one-vs-rest ROC (``pr``: recall, precision and AP) of ``f``'s
    calibrated probability, each row at its weight ``w`` where ``f`` has one (a weighted fold population,
    :func:`_fold_w`), from the highest threshold down; None with one class only."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    y = f["class_code"].to_numpy(np.int64) - 1 == k
    w = f["w"].to_numpy(np.float64) if "w" in f else np.ones(y.size)
    if not 0 < w[y].sum() < w.sum():
        return None
    s, prev = f[f"p_c{k}_cal"].to_numpy(np.float64), float(w[y].sum() / w.sum())
    if pr:  # sklearn's last point (recall 0, precision 1) has no threshold: dropped, the rest reversed
        prec, rec, _ = precision_recall_curve(y, s, sample_weight=w)
        return rec[-2::-1], prec[-2::-1], float(average_precision_score(y, s, sample_weight=w)), prev
    fpr, tpr, _ = roc_curve(y, s, sample_weight=w, drop_intermediate=False)
    return fpr, tpr, float(roc_auc_score(y, s, sample_weight=w)), prev


def _ovr_grid(x: np.ndarray, y: np.ndarray, pr: bool) -> np.ndarray:
    """A curve on ``ROC_GRID``: the ROC's step-read TPR (:func:`tpr_at`), or the PR's interpolated precision (the best
    precision at recall >= r; recall is non-decreasing along the curve)."""
    if not pr:
        return tpr_at(x, y, ROC_GRID)
    best = np.maximum.accumulate(y[::-1])[::-1]
    return best[np.minimum(np.searchsorted(x, ROC_GRID, side="left"), x.size - 1)]


def _ovr_curves(T: Dict[str, Any], c: Dict[str, Any], *, pr: bool, split: str = "test", **_: Any) -> Any:
    """X4: rows = models; each class's one-vs-rest ROC (``pr``: precision-recall) of its calibrated probability and
    their macro average (the class curves averaged on ``ROC_GRID``): pooled OOF curve thick in the class colour with its
    AUC (AP) in the panel corner, thin per-fold curves (each on its weighted fold population, :func:`_fold_w`) and their
    min-max band; PR adds the prevalence baseline (dashed)."""
    fs, g, cols = _seam(), T["guids"], [f"p_c{k}_cal" for k in range(3)]
    g = _sel(g, split=split).dropna(subset=cols) if set(cols) <= set(g.columns) else g.iloc[:0]
    models = _models(g)
    if not models:
        return _all_empty(1, 4)
    colors = {**fs.group_colors(CLASSES), "macro": fs.INK}
    fig, grid, _ = _facets(len(models), 4, row_h=2.45, width=10.6, sharex=False)
    for row, (m, sd) in zip(grid, models):
        u = _sel(g, model_id=m, seed=sd)
        u = u.assign(unit=u["guid"], w=_fold_w(T, c, u, split))
        parts = [*((str(f), x) for f, x in u.groupby("fold")),
                 ("pooled", pool_rows(u, c["data"]["shared_test_policy"], split).assign(w=1.0))]
        curves = {(p, k): r for p, f in parts for k in range(3) for r in [_ovr_curve(f, k, pr)] if r is not None}
        grid_v = {key_: _ovr_grid(r[0], r[1], pr) for key_, r in curves.items()}
        for p, _f in parts:  # macro: the three class curves averaged on the grid
            if all((p, k) in grid_v for k in range(3)):
                grid_v[(p, 3)] = np.mean([grid_v[(p, k)] for k in range(3)], axis=0)
                curves[(p, 3)] = (ROC_GRID, grid_v[(p, 3)], float(np.mean([curves[(p, k)][2] for k in range(3)])), np.nan)
        for k, (ax, name) in enumerate(zip(row, (*CLASSES, "macro"))):
            per = [(curves[(p, k)], grid_v[(p, k)]) for p, _f in parts[:-1] if (p, k) in curves]
            for i, (r, _v) in enumerate(per):
                ax.plot(r[0], r[1], color=colors[name], lw=fs.LINE_THIN, alpha=0.3, label="per fold" if i == 0 else None)
            if per:
                V = np.array([v for _r, v in per])
                ax.fill_between(ROC_GRID, V.min(0), V.max(0), color=colors[name], alpha=0.12, lw=0)
            r = curves.get(("pooled", k))
            if r is not None:
                ax.plot(r[0], r[1], color=colors[name], lw=fs.LINE_EMPHASIS * 2, label="pooled", zorder=3)
                _stat(ax, f"{'AP' if pr else 'AUC'} {r[2]:.3f}", "lower left" if pr else "lower right")
                if pr and k < 3:
                    ax.axhline(r[3], ls=(0, (3, 3)), color=colors[name], lw=fs.LINE_HAIRLINE, label="prevalence")
            if not ax.lines:
                _empty(ax)
                continue
            _unit_square(ax, diagonal=not pr)
            ax.set(xlabel="recall" if pr else "FPR", ylabel=("precision" if pr else "sensitivity") if k == 0 else None)
            fs.style_axes(ax)
        _row_label(row[-1], _who(m, sd))
    for ax, name in zip(grid[0], (*CLASSES, "macro")):
        ax.set_title(f"{name} vs rest" if name != "macro" else "Macro average")
    _tag(fig, f"One-vs-rest {'precision-recall' if pr else 'ROC'}", split=split)
    fs.caveat_note(fig, text=f"Thin: per fold; band: fold min-max. {_fold_phrase(T, c, split)}".rstrip())
    return fig


def _per_class_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                       fold: Optional[str] = None, **_: Any) -> Any:
    """X5: rows = models; (1) each class's argmax recall over the bin-present GUIDs' snapshots, then (2-4) under each
    metric type the primary policy's one-vs-rest sensitivity (solid ○) and FPR (dashed △) of each class with its own
    OvR threshold (specificity = 1 - FPR is in the tables); the primary model's per-class n strip under (1); one shared
    x axis."""
    from matplotlib.lines import Line2D

    fs, pid, mn = _seam(), c["eval"]["primary_policy"], c["eval"]["min_bin_class_n"]
    pol, d = _policy(c, pid), _x_time_rows(T, axis, split)
    models = _models(_sel(d, policy_id="argmax"))
    if not models:
        return _all_empty()
    colors = fs.group_colors(CLASSES)
    fig, grid, strips = _facets(len(models), 4, strip=(0,), width=fs.WINDOWS_FIGURE_WIDTH)
    _xlim(fig.axes, d["t"], axis, pol)
    for rows, (m, sd) in zip(grid, models):
        x = _sel(d, model_id=m, seed=sd)
        wc = _argmax_wide(_sel(x, policy_id="argmax"))
        for k, name in enumerate(CLASSES):
            _series(rows[0], _argmax_series(wc, f"recall_c{k}", mn), c, color=colors[name], marker="o", label=None,
                    fold=fold)
        for ax, mt in zip(rows[1:4], TYPES):
            for k, name in enumerate(CLASSES):
                r = _rates(_sel(_ovr_rows(x, k, OVR_RATES), metric_type=mt), pid)
                _series(ax, r["sens"], c, color=colors[name], marker="o", label=None, fold=fold)
                _series(ax, r["fpr"], c, color=colors[name], marker="^", label=None, fold=fold)
        rows[0].set(ylabel="rate")
        for ax in rows[:4]:
            if not ax.lines:
                _empty(ax)
                continue
            if ax is not rows[0] and pol.get("alpha"):
                ax.axhline(pol["alpha"], ls=(0, (3, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
            _unit_y(ax)
            _time_lines(ax, axis, pol)
            fs.style_axes(ax)
        _row_label(rows[3], _who(m, sd))
    fs.add_key(fig, [*(Line2D([], [], color=colors[n], lw=2) for n in CLASSES),
                     Line2D([], [], color=fs.MUTED, marker="o", ms=4.5, mec=fs.EDGE, mew=fs.MARKER_EDGE),
                     Line2D([], [], color=fs.MUTED, marker="^", ms=4.5, mec=fs.EDGE, mew=fs.MARKER_EDGE)],
               [*CLASSES, "recall / sensitivity", "FPR"])
    for ax, title in zip(grid[0], ("Argmax recall", *(f"OvR · {TYPE_LABEL[mt].replace(chr(10), ' ').lower()}"
                                                      for mt in TYPES))):
        ax.set_title(title)
    sm = _strip_model(models)
    _class_strip(strips[0], _argmax_wide(_sel(d, model_id=sm[0], seed=sm[1], policy_id="argmax")), c, fold)
    _time_lines(strips[0], axis, pol)
    for j in range(4):
        _x_time(_bottom(grid, strips, j), axis)
    _tag(fig, f"Per-class rates over time · policy {pid}", split=split, fold=fold)
    _under_note(fig, c, f"n strip: {_who(*sm)}.", _fold_phrase(T, c, split))
    return fig


def _per_class_auroc_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test",
                             fold: Optional[str] = None, **_: Any) -> Any:
    """X6: per model, each class's one-vs-rest snapshot AUROC (bin-present GUIDs, snapshot ``logit p_c<k>_cal``) vs
    time with its cluster-bootstrap band; the primary model's per-class n strip under each column; NaN where a class
    has too few GUIDs."""
    fs, d, key = _seam(), _x_time_rows(T, axis, split), _Once()
    ovr = [_ovr_rows(d, k, ("auroc",)) for k in range(3)]
    models = sorted(set().union(*(_models(o) for o in ovr)))
    if not models:
        return _all_empty()
    colors = fs.group_colors(CLASSES)
    fig, axes, strips = _model_panels(models)
    _xlim(fig.axes, d["t"], axis)
    for a, (m, sd) in zip(axes, models):
        for k, name in enumerate(CLASSES):
            f = _sel(ovr[k], model_id=m, seed=sd, metric_type="threshold_free", denominator="bin_present", metric="auroc")
            _series(a, f, c, color=colors[name], marker="o", label=f"{name} vs rest", fold=fold)
        if a.lines:
            a.axhline(0.5, ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE)
            _unit_y(a)
            fs.style_axes(a)
        else:
            _empty(a)
        a.set(ylabel="snapshot AUROC", title=_who(m, sd))
        _time_lines(a, axis)
        key(a)
    sm = _strip_model(models)
    wc = _argmax_wide(_sel(d, model_id=sm[0], seed=sm[1], policy_id="argmax"))
    for j, s in enumerate(strips):
        _class_strip(s, wc, c, fold, legend=j == 0)
        _time_lines(s, axis)
        _x_time(s, axis)
    _tag(fig, "Per-class AUROC over time", split=split, fold=fold)
    fs.caveat_note(fig, text=f"Hollow, black ring: no estimate (fewer than {c['eval']['min_bin_class_n']} GUIDs of a class in the bin; "
                             "the line is joined over it). "
                             f"n strip: {_who(*sm)}. {_fold_phrase(T, c, split)}".rstrip())
    return fig


def _f1_vs_time(T: Dict[str, Any], c: Dict[str, Any], *, axis: str, split: str = "test", fold: Optional[str] = None,
                **_: Any) -> Any:
    """X7: rows = models; (1) the argmax snapshot's (one GUID per bin) top-1 accuracy, macro and weighted F1, (2) each
    class's F1; the primary model's per-class n strip under (1); hollow where a class has fewer than
    ``eval.min_bin_class_n`` GUIDs."""
    fs, key_a, key_b = _seam(), _Once(), _Once()
    d = _sel(_x_time_rows(T, axis, split), policy_id="argmax")
    mn = c["eval"]["min_bin_class_n"]
    models = _models(d)
    if not models:
        return _all_empty()
    colors = fs.group_colors(CLASSES)
    fig, grid, strips = _facets(len(models), 2, strip=(0,), width=10.0)
    _xlim(fig.axes, d["t"], axis)
    for (a, b), (m, sd) in zip(grid, models):
        wc = _argmax_wide(_sel(d, model_id=m, seed=sd))
        for name, label, color, marker in (("top1_acc", "top-1 accuracy", fs.BLUE, "o"),
                                           ("macro_f1", "macro F1", fs.ORANGE, "s"),
                                           ("weighted_f1", "weighted F1", fs.VIOLET, "D")):
            _series(a, _argmax_series(wc, name, mn), c, color=color, marker=marker, label=label, fold=fold)
        for k, name in enumerate(CLASSES):
            _series(b, _argmax_series(wc, f"f1_c{k}", mn), c, color=colors[name], marker="o", label=f"{name} F1",
                    fold=fold)
        for ax, key in ((a, key_a), (b, key_b)):
            ax.set(ylabel="score" if ax is a else None)
            if not ax.lines:
                _empty(ax)
                continue
            _unit_y(ax)
            _time_lines(ax, axis)
            key(ax)
            fs.style_axes(ax)
        _row_label(b, _who(m, sd))
    grid[0, 0].set_title("Accuracy and F1")
    grid[0, 1].set_title("Per-class F1")
    sm = _strip_model(models)
    _class_strip(strips[0], _argmax_wide(_sel(d, model_id=sm[0], seed=sm[1])), c, fold)
    _time_lines(strips[0], axis)
    for j in range(2):
        _x_time(_bottom(grid, strips, j), axis)
    _tag(fig, "Argmax scores over time", split=split, fold=fold)
    _under_note(fig, c, f"n strip: {_who(*sm)}.", _fold_phrase(T, c, split))
    return fig


def _collapse_vs_binary(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", **_: Any) -> Any:
    """X9: per multi-task model, the percentile rank of the binary head's GUID score against that of the aux 3-class
    collapse (pooled OOF GUIDs, coloured by class; ranks, since both AUROCs and the Spearman correlation are rank
    statistics and a calibrated logit can span orders of magnitude), with the pooled AUROC of each, their Spearman
    correlation and the disagreement at the primary policy (``collapse/*`` rows; the CIs are in ``summary.md``)."""
    from teb_vae.classifier.config import TASKS

    fs, g, lab, key = _seam(), T["guids"], c["labels"], _Once()
    cols = [f"p_c{k}" for k in range(3)]
    g = _sel(g, split=split).dropna(subset=cols) if set(cols) <= set(g.columns) and TASKS.get(lab["task"]) else g.iloc[:0]
    models = _models(g)
    if not models:
        return _all_empty()
    met = _sel(T["metrics"], split=split, fold="pooled", level="guid")
    met = met[met["metric"].astype(str).str.startswith("collapse/")] if len(met) else met
    colors = fs.group_colors(CLASSES)
    fig, axes = _panels(len(models), 3, 2.7)
    for ax, (m, sd) in zip(axes, models):
        u = _sel(g, model_id=m, seed=sd)
        u = pool_rows(u.assign(unit=u["guid"]), c["data"]["shared_test_policy"], split)
        xs = pd.Series(aux_collapse(u, lab["task"])).rank(pct=True).to_numpy()
        ys = u["score_final_cal"].rank(pct=True).to_numpy()
        ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color=fs.FAINT, lw=fs.LINE_HAIRLINE)
        for name in _order(u["clinical_class"].unique(), "clinical_class"):
            s = (u["clinical_class"] == name).to_numpy(bool)
            ax.plot(xs[s], ys[s], "o", ms=4, alpha=0.75, color=colors.get(name, fs.FAINT), mec=figstyle.EDGE, mew=0.4,
                    label=name)
        v = _sel(met, model_id=m, seed=sd).drop_duplicates("metric").set_index("metric") if len(met) else met
        _stat(ax, "\n".join(f"{k} {_value(v, f'collapse/{n}')}" for k, n in (
            ("AUROC binary", "auroc_binary"), ("AUROC aux", "auroc_aux"), ("Spearman", "spearman"),
            ("disagreement", "disagreement"))), "upper left")
        ax.set(xlabel="aux collapse (rank)", ylabel="binary head (rank)", title=_who(m, sd))
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_aspect("equal", adjustable="box")
        key(ax)
        fs.style_axes(ax)
    _tag(fig, "Binary head vs aux 3-class collapse", split=split)
    return fig


_BUILDERS.update({CONFUSION_BINARY: _confusion_binary, CONFUSION_3CLASS: _confusion_3class,
                  ROC_OVR: partial(_ovr_curves, pr=False), PR_OVR: partial(_ovr_curves, pr=True),
                  COLLAPSE_VS_BINARY: _collapse_vs_binary})
AXIS_BUILDERS.update({CONFUSION_EVOLUTION: _confusion_evolution, PER_CLASS_VS_TIME: _per_class_vs_time,
                      PER_CLASS_AUROC_VS_TIME: _per_class_auroc_vs_time, F1_VS_TIME: _f1_vs_time})
EXPECTED_WHEN.update({s: _is_three_class for s in (CONFUSION_3CLASS, ROC_OVR, PR_OVR, CONFUSION_EVOLUTION,
                                                   PER_CLASS_VS_TIME, PER_CLASS_AUROC_VS_TIME, F1_VS_TIME)})
EXPECTED_WHEN[COLLAPSE_VS_BINARY] = _is_multi_task
CORE_EXTRA.append(CONFUSION_BINARY)

# ---- block E (errors and interpretation: E1-E5) ----
# E3-E5 are drawn from ``error_segments.parquet`` (the primary model's pooled test segments, each GUID once; GUID values
# via :func:`_e_guids`), the §10 top-errors table from ``errors.parquet``. The E2 pages are not registry stems (their
# GUIDs depend on the data): :func:`render_pages` draws them in ``report`` (one fail-soft step) under
# ``evaluation/pages/fold_<k>/<guid>`` with the ``pages.csv`` manifest, from the prediction tables and the frozen
# feature cache only. E6 (``attribution_channels``, only when ``eval.attribution.enabled``) is drawn from ``attribution.parquet``.
from teb_vae.classifier.metrics import primary_model  # noqa: E402

SCORE_VS_QUALITY = "errors/score_vs_quality"
SCORE_VS_LENGTH = "errors/score_vs_length"
ATTENTION_SUMMARY = "errors/attention_summary"
ATTRIBUTION_CHANNELS = "errors/attribution_channels"
PAGES_CSV = "pages.csv"
#: Poolings whose step weights are learned (``mean``/``mean_max`` weigh the valid steps uniformly).
ATTENTION_POOLINGS = ("gated_attention", "query", "conjunctive")
#: Decades of dynamic range a log colour scale or axis keeps below its maximum (``traces.LOG_PANEL_DECADES``).
LOG_DECADES = 4.0
ROW_INCHES = 1.0
#: The note under an E2 page, where the feature cache records a causal group delay (the page's one caveat).
PAGE_CAVEAT = "Input coefficients are causal: group delay up to {} s per channel."


def _has_attention(c: Dict[str, Any]) -> bool:
    """E5 is expected when the model has learned attention: a weighted pooling or an ``attention_mil`` sequence."""
    m = c.get("model") or {}
    return ((m.get("pooling") or {}).get("kind") in ATTENTION_POOLINGS
            or (m.get("scope") == "sequence" and (m.get("sequence") or {}).get("kind") == "attention_mil"))


# Ported from teb_vae/lag_attn_cfs/eval/attributions.py:849-927 (masked_field, signed_log_norm, unsigned_log_norm,
# symlog_axis): that module imports captum at load time. The unused ``anchor`` / ``headroom`` options are dropped.
def masked_field(field: np.ndarray, live: Optional[np.ndarray]) -> np.ndarray:
    """A float64 copy of a (T, C) map with NaN where step t < ``live[c]`` (cells the model never read), drawn as the
    bad colour so a blank cell reads as "not read" and never sets a colour scale."""
    out = np.array(field, dtype=np.float64, copy=True)
    if live is not None:
        out[np.arange(out.shape[0])[:, None] < np.asarray(live, dtype=np.int64)[None, :out.shape[1]]] = np.nan
    return out


def signed_log_norm(field: np.ndarray) -> Optional[Any]:
    """A symmetric-log colour scale about zero, linear within max|a|·10^-LOG_DECADES; None on an empty field."""
    from matplotlib import colors as mcolors

    finite = np.asarray(field, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size or not np.abs(finite).max() > 0.0:
        return None
    limit = float(np.abs(finite).max())
    return mcolors.SymLogNorm(linthresh=limit * 10.0 ** (-LOG_DECADES), vmin=-limit, vmax=limit, base=10)


def unsigned_log_norm(field: np.ndarray) -> Optional[Any]:
    """A log colour scale over a non-negative field's positive mass, or None when it has none."""
    from matplotlib import colors as mcolors

    finite = np.asarray(field, dtype=np.float64)
    positive = finite[np.isfinite(finite) & (finite > 0.0)]
    if not positive.size:
        return None
    top = float(positive.max())
    return mcolors.LogNorm(vmin=top * 10.0 ** (-LOG_DECADES), vmax=top)


def log_ticks(top: float, floor: float, signed: bool) -> List[float]:
    """Every second decade from the decade of ``top`` down to ``floor`` (and its mirror and 0 when ``signed``): the
    ticks of a log or symmetric-log scale over LOG_DECADES, few enough that their labels never overlap."""
    if not (top > 0 and floor > 0):
        return []
    pos = 10.0 ** np.arange(np.floor(np.log10(top)), np.log10(floor) - 1e-9, -2.0)
    return sorted([*(-pos if signed else []), *([0.0] if signed else []), *pos])


def symlog_axis(ax: Any, *values: Any, axis: str = "y") -> None:
    """A symmetric-log axis floored LOG_DECADES below the data's largest magnitude, ticked at every second decade
    (:func:`log_ticks`); linear when the data has no magnitude or fewer than three such ticks fall in view."""
    from matplotlib.ticker import FixedLocator, NullLocator

    stacked = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values]) if values else np.zeros(0)
    finite = stacked[np.isfinite(stacked)]
    if not finite.size or not np.abs(finite).max() > 0.0:
        return
    threshold = float(np.abs(finite).max()) * 10.0 ** (-LOG_DECADES)
    (ax.set_yscale if axis == "y" else ax.set_xscale)("symlog", linthresh=threshold, base=10)
    ax.autoscale_view()  # the view interval is stale until the artists drawn so far are autoscaled
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = sorted(a.get_view_interval())
    ticks = [t for t in log_ticks(max(abs(lo), abs(hi)), threshold, True) if lo <= t <= hi]
    if len(ticks) >= 3:
        a.set_major_locator(FixedLocator(ticks))
        a.set_minor_locator(NullLocator())
    else:  # data within about one decade: a linear axis reads better than a few unlabelled log ticks
        (ax.set_yscale if axis == "y" else ax.set_xscale)("linear")


def _e_rows(T: Dict[str, Any], split: str, fold: Optional[str]) -> pd.DataFrame:
    """``error_segments`` rows of ``split`` (and one fold's GUIDs); empty when the table is."""
    s = T["error_segments"]
    if s.empty or "alarm_score" not in s:
        return s.iloc[:0]
    return s[(s["split"] == split) & ((s["fold"].astype(str) == str(fold)) if fold is not None else True)]


def _e_guids(T: Dict[str, Any], split: str = "test", fold: Optional[str] = None) -> pd.DataFrame:
    """One row per GUID of :func:`_e_rows`: ``y``, ``clinical_class``, alarm score ``max_score`` (the running max at
    its last segment), ``alarmed`` (primary policy), ``n_segments``, ``span_h`` and the mean ``valid_frac``."""
    s = _e_rows(T, split, fold)
    if s.empty:
        return pd.DataFrame(columns=["fold", "guid", "y", "clinical_class", "max_score", "alarmed", "n_segments",
                                     "valid_frac", "span_h"])
    g = s.sort_values(["guid", "seg_pos"]).groupby(["fold", "guid"], as_index=False).agg(
        y=("y", "first"), clinical_class=("clinical_class", "first"), max_score=("alarm_score", "last"),
        alarmed=("alarmed", "first"), n_segments=("seg_pos", "size"), valid_frac=("valid_frac", "mean"),
        start=("epoch_s", "min"), end=("t_end_s", "max"))
    return g.assign(span_h=(g["end"] - g["start"]) / 3600.0)


def _e_who(T: Dict[str, Any]) -> str:
    """The primary model of the E figures, for their title."""
    s = T["error_segments"]
    return f"{s['model_id'].iloc[0]} (seed {s['seed'].iloc[0]})" if len(s) and "model_id" in s else "no model"


def _by_class(ax: Any, g: pd.DataFrame, x: str, xlabel: str, title: str, legend: bool = True) -> None:
    """Scatter of the alarm score against ``x`` per clinical class, on a symlog axis."""
    fs = _seam()
    g = g.dropna(subset=[x, "max_score"])
    if g.empty:
        return _empty(ax)
    order = _order(g["clinical_class"].unique(), "clinical_class")
    colors = fs.group_colors(order)
    for k in order:
        v = g[g["clinical_class"] == k]
        ax.scatter(v[x], v["max_score"], s=16, color=colors[k], alpha=0.75, edgecolors="white", linewidths=0.4,
                   label=k)
    symlog_axis(ax, g["max_score"])
    ax.set(xlabel=xlabel, ylabel="alarm score (logit)", title=title)
    if legend:
        _key(ax)
    fs.style_axes(ax)


def _qbins(x: Any, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(codes, edges)``: quantile bins of ``x`` (q equal-mass bins, duplicate edges merged; a bin is (lo, hi],
    the first [lo, hi], as block S's tertiles), one bin for constant data."""
    x = np.asarray(x, np.float64)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, q + 1))) if x.size else np.zeros(1)
    return np.searchsorted(edges[1:-1], x, side="left"), edges


def _rate_bars(ax: Any, g: pd.DataFrame, codes: np.ndarray, edges: np.ndarray, kinds: tuple, title: str,
               legend: bool = True) -> None:
    """Per quantile bin of the GUIDs' ``valid_frac``: the FP rate (alarmed healthy / healthy) and FN rate (adverse not
    alarmed / adverse) of ``kinds`` under the primary policy, with Wilson 95% bars and N per bin."""
    from teb_vae.classifier.metrics import wilson

    fs = _seam()
    if g.empty:
        return _empty(ax)
    x, width = np.arange(max(len(edges) - 1, 1)), 0.8 / len(kinds)
    for i, (name, cls, hit, color) in enumerate(kinds):
        sel = (g["y"] == cls).to_numpy()
        n = np.bincount(codes[sel], minlength=x.size)
        k = np.bincount(codes[sel], weights=(g["alarmed"].to_numpy(bool) == hit)[sel], minlength=x.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = k / n
        lo, hi = wilson(k, n)
        ax.bar(x + (i - (len(kinds) - 1) / 2) * width, r, width, color=color, alpha=0.85, edgecolor="white", lw=0.6,
               yerr=np.vstack([r - lo, hi - r]), error_kw={"lw": fs.LINE_REGULAR, "ecolor": fs.INK}, label=name)
    ax.set_xticks(x, [f"≤ {e:.3g}" for e in edges[1:]] if len(edges) > 1 else [f"= {edges[0]:.3g}"], rotation=30,
                  ha="right")
    ax.set(ylim=(0, 1.05), ylabel="rate", xlabel="valid_frac (bin upper edge)", title=title)
    if legend:
        _key(ax)
    fs.style_axes(ax, grid="y")


def _score_vs_quality(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                      **_: Any) -> Any:
    """E3: (a) GUID alarm score vs mean ``valid_frac`` by class; (b) FP and FN rate per ``valid_frac`` decile and
    (c) the FP rate per quality tertile, at the primary policy (Wilson 95%)."""
    fs = _seam()
    g = _e_guids(T, split, fold)
    fig, axes = _figure(1, 3, 2.8)
    healthy, adverse = fs.CLINICAL_CLASS_COLORS["healthy"], fs.CLINICAL_CLASS_COLORS["hie"]
    _by_class(axes[0, 0], g, "valid_frac", "mean valid_frac", "Alarm score vs signal quality")
    _rate_bars(axes[0, 1], g, *_qbins(g["valid_frac"], 10), (("FP rate, healthy", 0, True, healthy),
                                                            ("FN rate, adverse", 1, False, adverse)),
               "Error rate per valid_frac decile")
    h = g[g["y"] == 0]
    _rate_bars(axes[0, 2], h, *_qbins(h["valid_frac"], 3), (("FP rate, healthy", 0, True, healthy),),
               "FP rate per quality tertile", legend=False)
    _tag(fig, f"Errors vs signal quality · {_e_who(T)}", split=split, fold=fold)
    return fig


def _score_vs_length(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                     **_: Any) -> Any:
    """E4: GUID alarm score vs the number of segments and vs the recording span, by class (the length shortcut)."""
    g = _e_guids(T, split, fold)
    fig, axes = _figure(1, 2, 2.8)
    _by_class(axes[0, 0], g, "n_segments", "segments per GUID", "Alarm score vs length")
    _by_class(axes[0, 1], g, "span_h", "recording span (h)", "Alarm score vs span", legend=False)
    _tag(fig, f"Errors vs recording length · {_e_who(T)}", split=split, fold=fold)
    return fig


def _attention_summary(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: Optional[str] = None,
                       **_: Any) -> Any:
    """E5: per segment by class, (a) the pooling attention mass on the late half of the post-warm-up span and (b) its
    centroid (0 = the warm-up boundary, 1 = the last valid step); (c) the sequence attention the final position gives
    each segment vs hours before delivery (``attention_mil``), one line per GUID."""
    fs = _seam()
    s = _e_rows(T, split, fold)
    fig, axes = _figure(1, 3, 2.8)
    order = _order(s["clinical_class"].unique(), "clinical_class") if len(s) else []
    colors = fs.group_colors(order) if order else {}
    for ax, col, title in ((axes[0, 0], "attn_late_mass", "Step attention on the late half"),
                           (axes[0, 1], "attn_centroid", "Step attention centroid")):
        v = s.dropna(subset=[col]) if col in s else s.iloc[:0]
        fs.violin_panel(ax, {f"{k} (N = {int((v['clinical_class'] == k).sum())})": v.loc[v["clinical_class"] == k, col]
                             for k in order}, title=title, ylabel="per segment, 0-1",
                        colors={f"{k} (N = {int((v['clinical_class'] == k).sum())})": colors[k] for k in order})
        ax.axhline(0.5, ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        fs.style_axes(ax, grid="y")
    ax = axes[0, 2]
    ax.set_title("Final-position attention")
    v = s.dropna(subset=["seq_attn_final"]) if "seq_attn_final" in s else s.iloc[:0]
    if v.empty:
        _empty(ax)
    else:
        for (k, _), x in v.sort_values("seg_pos").groupby(["clinical_class", "guid"]):
            ax.plot(-x["t_end_s"] / 3600.0, x["seq_attn_final"], color=colors[k], lw=fs.LINE_THIN, alpha=0.35)
        for k in order:
            ax.plot([], [], color=colors[k], lw=2, label=k)
        ax.set_ylabel("attention weight")
        _x_time(ax, "to_delivery")
        _key(ax)
        fs.style_axes(ax)
    _tag(fig, f"Attention · {_e_who(T)}", split=split, fold=fold)
    return fig


# ---- E2 trajectory pages ----
def per_class_rows(frame: pd.DataFrame, *, per_class: int, seed: int) -> pd.DataFrame:
    """Up to ``per_class`` rows drawn uniformly at random from every clinical class (equal N, not a proportional
    quota), classes worst first, returned in table order. Ported from ``teb_vae/lag_attn_cfs/eval/analyses/samples.py:
    691 per_class_rows`` (analysis modules are not imported); class i draws with ``subsample_indices`` at seed + i."""
    from teb_vae.lag_attn.eval.masks import subsample_indices

    if frame.empty or "clinical_class" not in frame.columns:
        return frame.head(0)
    cls, picked = frame["clinical_class"].to_numpy(), []
    for i, name in enumerate(_order(pd.unique(frame["clinical_class"].dropna()), "clinical_class")):
        members = np.flatnonzero(cls == name)
        drawn = subsample_indices(len(members), int(per_class), int(seed) + i)
        picked += list(members if drawn is None else members[drawn.numpy()])
    return frame.iloc[sorted(picked)]


def page_selection(guids: pd.DataFrame, errors: pd.DataFrame, *, per_class: int, top_errors: int,
                   seed: int) -> pd.DataFrame:
    """E2 pages of one model and split: per fold, :func:`per_class_rows` of its GUIDs (seed + fold) plus the E1 GUIDs
    of rank <= ``top_errors`` (fp and fn). One row per (fold, guid) with ``reason`` ("sample", "fp1", ..., joined by
    "+"), in fold, then table order."""
    parts = []
    for fold, g in guids.groupby("fold", sort=True):
        parts.append(per_class_rows(g, per_class=per_class, seed=seed + int(fold)).assign(reason="sample"))
        e = errors[(errors["fold"] == fold) & (errors["rank"] <= top_errors)] if len(errors) else errors
        parts.append(e.assign(reason=e["kind"].astype(str) + e["rank"].astype(str)) if len(e) else e)
    cols = ["fold", "guid", "clinical_class", "reason"]
    x = pd.concat([p.reindex(columns=cols) for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=cols)
    return x.groupby(["fold", "guid"], sort=False, as_index=False).agg(
        clinical_class=("clinical_class", "first"), reason=("reason", "+".join))


def _feature_groups(channels: Any) -> List[Tuple[str, slice]]:
    """Contiguous channel groups by name stem (``fhr_st[3]`` -> ``fhr_st``), in channel order."""
    stems, out = [str(ch).split("[")[0] for ch in channels], []
    for i, name in enumerate(stems):
        if out and out[-1][0] == name:
            out[-1] = (name, slice(out[-1][1].start, i + 1))
        else:
            out.append((name, slice(i, i + 1)))
    return out


def _map_row(ax: Any, cax: Any, X: np.ndarray, mask: np.ndarray, t0_s: np.ndarray, step_s: float, name: str) -> None:
    """One channel group's input map: each segment's valid steps placed at their stored time (hours before delivery),
    masked and cold cells blank (never drawn, never scaled), on a symlog (signed) or log (non-negative) colour scale."""
    import matplotlib

    fs, fields = _seam(), []
    for i in range(len(X)):
        m = mask[i]
        if not m.any():
            continue
        on = m[:, None] & (X[i] != 0)
        # ponytail: the cache keeps no per-channel warm-up, so a channel's leading exact zeros on valid steps are read
        # as the source's zeroed cold cells; store causal_warmup_steps in the fingerprint to read them instead
        F = masked_field(X[i], np.where(on.any(0), on.argmax(0), len(m)))
        F[~m] = np.nan
        lo, hi = np.flatnonzero(m)[[0, -1]]
        fields.append((-(t0_s[i] + step_s * lo) / 3600.0, -(t0_s[i] + step_s * (hi + 1)) / 3600.0, F[lo:hi + 1]))
    vals = np.concatenate([f.ravel() for *_, f in fields]) if fields else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    signed = bool((vals < 0).any())
    norm = signed_log_norm(vals) if signed else unsigned_log_norm(vals)
    ax.set_title(f"Input {name}")
    if norm is None:
        cax.set_visible(False)
        return _empty(ax)
    cmap = matplotlib.colormaps["RdBu_r" if signed else "viridis"].with_extremes(bad=(0, 0, 0, 0))
    for x0, x1, F in fields:
        im = ax.imshow(F.T, aspect="auto", origin="lower", extent=(x0, x1, -0.5, F.shape[1] - 0.5), cmap=cmap,
                       norm=norm, interpolation="nearest")
    ax.set(ylim=(-0.5, X.shape[-1] - 0.5), ylabel="channel")
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_ticks(log_ticks(norm.vmax, norm.linthresh if signed else norm.vmin, signed))
    cbar.minorticks_off()
    cbar.set_label("symlog" if signed else "log", fontsize=fs.FONT_SMALL)
    cbar.ax.tick_params(labelsize=fs.FONT_TINY)
    cbar.outline.set_visible(False)


def _trajectory_page(g: pd.DataFrame, feats: Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]], c: Dict[str, Any],
                     *, thresholds: Dict[str, float], geometry: Tuple[float, float], header: str, caveat: str = "") -> Any:
    """One E2 page: rows stacked on one time axis (hours before delivery, delivery on the right), the samples-page
    layout (14 in wide, a thin colour-bar column, header and footer strips).

    ``g``: the GUID's prediction segments (``epoch_s``, ``t_end_s``, ``seg_pos``, calibrated logits, attention scalars);
    ``feats``: ``(values (n, T', C), step_mask (n, T'), channels)`` aligned with ``g``'s rows, or None without the
    cache; ``thresholds``: the fold's GUID-level threshold per policy; ``geometry``: ``(step seconds, trim offset s)``.
    Rows: the online risk (calibrated online score, running max, the thresholds and the primary policy's first alarm),
    the segment-local scores over each segment's observed window, the pooling-attention scalars when present, and one
    input map per channel group. ``header`` is the one line naming the recording (GUID, subgroup, class), the model and
    the fold; ``caveat`` the optional one-line note under the page.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from teb_vae.classifier.thresholds import _online

    fs, pid = _seam(), c["eval"]["primary_policy"]
    o = _online(g)
    x_end = -o["t_end_s"].to_numpy(np.float64) / 3600.0
    attn = [a for a in ("attn_late_mass", "attn_centroid", "seq_attn_final") if a in o and o[a].notna().any()]
    groups = _feature_groups(feats[2]) if feats is not None else [("features", slice(0, 0))]
    specs = [("risk", 1.5), ("segments", 1.0), *([("attention", 0.8)] if attn else []), *(("map", 1.3) for _ in groups)]
    header_in, footer_in = 0.55, 0.9
    height = sum(h for _, h in specs) * ROW_INCHES + header_in + footer_in
    fig = plt.figure(figsize=(14, height))
    grid = GridSpec(len(specs), 2, figure=fig, height_ratios=[h for _, h in specs], width_ratios=[1.0, 0.018],
                    left=0.065, right=0.86, top=1 - header_in / height, bottom=footer_in / height, hspace=0.55, wspace=0.03)
    axes = []
    for i in range(len(specs)):
        axes.append((fig.add_subplot(grid[i, 0], sharex=axes[0][0] if axes else None), fig.add_subplot(grid[i, 1])))
    for ax, cax in axes[:len(specs) - len(groups)]:
        cax.set_visible(False)

    ax = axes[0][0]
    s, r = o["s"].to_numpy(np.float64), o["r"].to_numpy(np.float64)
    ax.plot(x_end, s, marker="o", ms=fs.MARKER_SMALL, lw=fs.LINE_REGULAR, color=fs.BLUE, mec=figstyle.EDGE, mew=0.4,
            label="online score")
    ax.plot(x_end, r, drawstyle="steps-post", lw=fs.LINE_EMPHASIS * 2, color=fs.INK, label="running max")
    for p, thr in thresholds.items():
        ax.axhline(thr, ls=(0, (4, 2)), color=_policy_color(c, p),
                   lw=fs.LINE_EMPHASIS * 1.6 if p == pid else fs.LINE_THIN, label=f"{p} threshold")
    hit = np.flatnonzero(r > thresholds.get(pid, np.inf))
    if hit.size:
        ax.axvline(x_end[hit[0]], ls=":", color=fs.ROSE, lw=fs.LINE_HEAVY * 0.7, label="first alarm")
    symlog_axis(ax, s, r, list(thresholds.values()))
    ax.set(title="Online risk", ylabel="logit (symlog)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fs.FONT_SMALL)

    ax = axes[1][0]
    seg = o["logit_seg_cal"].to_numpy(np.float64)
    if np.isfinite(seg).any():
        e = o["epoch_s"].to_numpy(np.float64)
        ax.hlines(seg, -(e + geometry[1]) / 3600.0, x_end, color=fs.TEAL, lw=fs.LINE_EMPHASIS * 2.4, alpha=0.8)
        ax.plot(x_end, seg, "o", ms=fs.MARKER_SMALL, color=fs.TEAL, mec=figstyle.EDGE, mew=0.4)
        symlog_axis(ax, seg)
        ax.set(title="Segment score", ylabel="logit (symlog)")
    else:
        _empty(ax)
        ax.set_title("Segment score")
    if attn:
        ax = axes[2][0]
        for a, marker, color in zip(attn, "os^", fs.LINE_PALETTE):
            ax.plot(x_end, o[a].to_numpy(np.float64), marker, ms=fs.MARKER_SMALL + 0.6, color=color, mec=figstyle.EDGE,
                    mew=0.4, label=a)
        ax.set(ylim=(-0.04, 1.04), title="Pooling attention", ylabel="0-1")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=fs.FONT_SMALL)

    maps = axes[len(specs) - len(groups):]
    if feats is None:
        _empty(maps[0][0])
        maps[0][0].set_title("Input features")
        maps[0][1].set_visible(False)
    else:
        t0 = o["epoch_s"].to_numpy(np.float64) + geometry[1]
        for (name, sl), (ax, cax) in zip(groups, maps):
            _map_row(ax, cax, feats[0][..., sl], feats[1], t0, geometry[0], name)
    for i, (ax, _) in enumerate(axes):
        fs.style_axes(ax, grid="x" if i >= len(specs) - len(groups) else "both")  # no grid lines across the maps
        ax.tick_params(labelbottom=False)
    first = min(float(np.nanmax(-o["epoch_s"].to_numpy(np.float64) / 3600.0)), 48.0)
    axes[0][0].set_xlim(first + 0.1, min(0.0, float(np.nanmin(x_end))) - 0.05)  # delivery on the right
    axes[-1][0].tick_params(labelbottom=True)
    axes[-1][0].set_xlabel("hours before delivery")
    fig.text(0.065, 1 - 0.14 / height, header, ha="left", va="top", fontsize=fs.FONT_TITLE, fontweight="bold",
             color=fs.INK)
    if caveat:
        fs.caveat_note(fig, text=caveat)
    fs.figures.mark_laid_out(fig)
    return fig


def _group_delay_s(fp: Dict[str, Any]) -> Optional[float]:
    """Largest ``causal_delay_s`` over the blocks of the first shard the cache was extracted from; None when none is
    recorded (a two-sided build) or the shard is unreadable."""
    import h5py

    try:
        with h5py.File(fp["shards"][0]["path"], "r") as h5:
            d = [float(np.max(h5[k].attrs["causal_delay_s"])) for k in h5 if "causal_delay_s" in h5[k].attrs]
    except (OSError, KeyError, IndexError, TypeError):
        return None
    return max(d) if d else None


def page_geometry(fp: Dict[str, Any]) -> Tuple[float, float]:
    """``(step seconds, trim offset seconds)`` of the cached features from the source fingerprint: ``time_pool`` and
    ``trim_minutes`` default to 1 only when unrecorded (a recorded trim of 0 is a 0-s offset)."""
    from teb_vae.classifier.cohort import SECONDS_PER_STEP

    pool, trim = fp.get("time_pool"), fp.get("trim_minutes")
    return SECONDS_PER_STEP * float(1 if pool is None else pool), 60.0 * float(1.0 if trim is None else trim)


def render_pages(run_dir: Path, T: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    """E2: :func:`page_selection` of the primary model's test GUIDs (``eval.trajectory_pages``; the model of
    ``error_segments``, none without one), one page each
    (:func:`_trajectory_page`) per ``eval.figure_formats`` under ``evaluation/pages/fold_<k>/<guid>``, and the
    ``pages.csv`` manifest (fold, guid, clinical_class, reason, model_id, seed, n_segments, path). Reads the prediction
    tables, ``thresholds.parquet``, ``errors.parquet`` and the frozen feature cache (``manifest.json`` source); pages
    are drawn without the input maps when the cache is gone."""
    from teb_vae.classifier.sources import read_rows

    fs, ev, tp = _seam(), c["eval"], c["eval"]["trajectory_pages"]
    out = run_dir / "evaluation" / "pages"
    out.mkdir(parents=True, exist_ok=True)
    es = T["error_segments"]  # the primary model with a per-position score (metrics.run_E), else no trajectory
    prim = (es["model_id"].iloc[0], es["seed"].iloc[0]) if len(es) and "model_id" in es else None
    cols = ["fold", "guid", "clinical_class", "reason", "model_id", "seed", "n_segments", "path"]
    if prim is None:
        pd.DataFrame(columns=cols).to_csv(out / PAGES_CSV, index=False)
        return {"n_pages": 0}
    m, sd = prim
    guids = _sel(T["guids"], model_id=m, seed=sd, split="test")
    sel = page_selection(guids, _sel(T["errors"], model_id=m, seed=sd, split="test"), per_class=tp["per_class"],
                         top_errors=tp["top_errors"], seed=int(c["run"]["seeds"][0]))
    seg = pd.read_parquet(run_dir / "predictions" / "segments.parquet",
                          filters=[("model_id", "==", m), ("seed", "==", sd), ("split", "==", "test")])
    seg = seg.merge(sel[["fold", "guid"]], on=["fold", "guid"]).sort_values(["fold", "guid", "seg_pos"])
    src = T["manifest"].get("source") or {}
    fp, cache = src.get("fingerprint") or {}, Path(str(src.get("cache_dir")))
    feats = None
    if (cache / "features.h5").is_file() and len(seg):
        index = pd.read_parquet(cache / "index.parquet", columns=["fold", "split", "guid", "epoch_s", "row"])
        rows = seg.merge(index, on=["fold", "split", "guid", "epoch_s"], how="left")["row"]
        got = read_rows(cache, rows.fillna(0).astype(int).to_numpy())
        mask = got.step_mask.numpy() & rows.notna().to_numpy()[:, None]
        X = got.values.numpy() if got.attn is None else np.concatenate([got.values.numpy(), got.attn.numpy()], -1)
        feats = (X, mask, got.channels)
    delay = _group_delay_s(fp)
    caveat = PAGE_CAVEAT.format(f"{delay:.0f}") if delay is not None else ""
    geometry = page_geometry(fp)
    thr = _sel(T["thr"], model_id=m, seed=sd, level="guid").reindex(columns=["fold", "policy_id", "threshold"])
    at = seg.reset_index(drop=True).groupby(["fold", "guid"], sort=False).indices
    manifest = []
    # ponytail: pages render serially here (~0.7 s each, ~150 at 10 folds); hand them to _render_all's pool if slow
    fs.configure_figure_style(ev["figure_formats"][0])
    for r in sel.itertuples(index=False):
        i = at.get((r.fold, r.guid))
        g = seg.iloc[i] if i is not None else seg.iloc[:0]
        if g.empty:
            continue
        th = {p: float(v) for p, v in thr.loc[thr["fold"].astype(str) == str(r.fold), ["policy_id", "threshold"]]
              .itertuples(index=False) if np.isfinite(v)}
        header = f"GUID {r.guid} · {g['subgroup'].iloc[0]} · {r.clinical_class} · {_who(m, sd)} · fold {r.fold}"
        path = out / f"fold_{r.fold}" / r.guid
        path.parent.mkdir(parents=True, exist_ok=True)
        f = None if feats is None else tuple(x[i] for x in feats[:2]) + (feats[2],)
        for fmt in ev["figure_formats"]:
            fs.figures.set_figure_format(fmt)
            fs.render_figure(_trajectory_page(g, f, c, thresholds=th, geometry=geometry, header=header, caveat=caveat),
                             path, crop=False)
        manifest.append({"fold": r.fold, "guid": r.guid, "clinical_class": r.clinical_class, "reason": r.reason,
                         "model_id": m, "seed": sd, "n_segments": len(g), "path": str(path.relative_to(out.parent))})
    pd.DataFrame(manifest, columns=cols).to_csv(out / PAGES_CSV, index=False)
    return {"n_pages": len(manifest), "model": f"{m}|{sd}", "dir": str(out)}


# ---- summary.md §10 (E1) ----
def _errors_md(T: Dict[str, Any], c: Dict[str, Any]) -> List[str]:
    """§10 top errors (E1): the primary model's test rows of rank <= ``eval.trajectory_pages.top_errors`` per fold and
    kind; the full table is ``tables/errors.parquet``."""
    e, k = T["errors"], c["eval"]["trajectory_pages"]["top_errors"]
    prim = primary_model([x for x in _models(e) if x[0] != "shuffled"]) if len(e) else None
    L = ["### Top errors (E1)", ""]
    if prim is None:
        return L + ["_no top-error rows (no model with a per-position score)_", ""]
    x = _sel(e, model_id=prim[0], seed=prim[1], split="test")
    x = x[x["rank"] <= k].sort_values(["fold", "kind", "rank"], ascending=[True, False, True])
    t = pd.DataFrame({"fold": x["fold"], "kind": x["kind"], "rank": x["rank"], "guid": x["guid"],
                      "class": x["clinical_class"], "subgroup": x["subgroup"], "stage (last)": x["stage"],
                      "alarm score": x["max_score"].map(_num3), "threshold": x["threshold"].map(_num3),
                      "alarmed": x["alarmed"], "first alarm (h before)": x["first_alarm_h"].map(_num3),
                      "segments": x["n_segments"], "span (h)": x["span_h"].map(_num3),
                      "valid_frac": x["valid_frac"].map(_num3), "has_tlo": x["has_tlo"]})
    return L + [f"`{prim[0]}` (seed {prim[1]}), test, primary policy `{c['eval']['primary_policy']}`: per fold the top "
                f"{k} false positives (healthy, highest alarm score) and false negatives (adverse, lowest) of "
                f"`eval.error_analysis.top_k` = {c['eval']['error_analysis']['top_k']} (each class's tail at most half "
                "the class). A listed GUID the policy decided correctly shows `alarmed` accordingly. Every model, val and "
                "test: `evaluation/tables/errors.parquet`; trajectory pages: `evaluation/pages/` (`pages.csv`).", "",
                _md(t) if len(t) else "_no rows_", ""]


def _attribution_channels(T: Dict[str, Any], c: Dict[str, Any], *, split: str = "test", fold: str = "pooled",
                          **_: Any) -> Any:
    """E6: per channel group and class, (a) the mean share of the GUID score's summed |IG| and (b) the mean signed IG
    (the group's net push on the logit), pooled test GUIDs of the primary model, 95% patient-cluster bootstrap CIs."""
    fs, a = _seam(), T["attribution"]
    a = a[(a["split"] == split) & (a["fold"].astype(str) == fold)] if len(a) else a
    fig, axes = _figure(1, 2, 2.8)
    groups = sorted(a["channel_group"].unique()) if len(a) else []
    order = _order(a["clinical_class"].unique(), "clinical_class") if len(a) else []
    colors = fs.group_colors(order) if order else {}
    key = _Once()
    for ax, met, title, ylabel in ((axes[0, 0], "share", "Share of |IG|", "mean share of |IG|"),
                                   (axes[0, 1], "signed", "Signed IG", "mean IG (logit)")):
        ax.set_title(title)
        v = a[a["metric"] == met] if len(a) else a
        if v.empty:
            _empty(ax)
            continue
        width = 0.8 / max(1, len(order))
        for i, k in enumerate(order):
            x = v[v["clinical_class"] == k].set_index("channel_group").reindex(groups)
            pos = np.arange(len(groups)) + (i - (len(order) - 1) / 2) * width
            err = np.vstack([x["value"] - x["ci_lo"], x["ci_hi"] - x["value"]]).clip(min=0)
            ax.bar(pos, x["value"], width, yerr=err, color=colors[k], edgecolor="white", lw=0.6,
                   error_kw={"lw": fs.LINE_REGULAR, "ecolor": fs.INK}, label=k)
        ax.set_xticks(np.arange(len(groups)), groups, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        key(ax)
        fs.style_axes(ax, grid="y")
    who = _who(a["model_id"].iloc[0], a["seed"].iloc[0]) if len(a) else "no model"
    _tag(fig, f"Attribution by channel group · {who}", split=split, fold=None if fold == "pooled" else fold)
    fs.caveat_note(fig, text="IG: integrated gradients of the GUID score (logit).")
    return fig


_BUILDERS.update({SCORE_VS_QUALITY: _score_vs_quality, SCORE_VS_LENGTH: _score_vs_length,
                  ATTENTION_SUMMARY: _attention_summary, ATTRIBUTION_CHANNELS: _attribution_channels})
EXPECTED_WHEN[ATTENTION_SUMMARY] = _has_attention
EXPECTED_WHEN[ATTRIBUTION_CHANNELS] = lambda c: bool(((c.get("eval") or {}).get("attribution") or {}).get("enabled"))
EXTRA_TABLES.extend(["errors", "error_segments", "attribution"])

# ---- block Q (model comparison: Q1-Q3) ----
# ``run.py compare`` output, written only under its ``--out``. The Q1 figures come from each compared run's evaluation
# tables (``roc_points``, ``metrics``: its primary model's pooled test rows), the Q2 forest from ``comparison.parquet``
# (``metrics.compare_runs``), and ``comparison.md`` is Q3. Every figure gives each run one colour (the line palette in
# run order, the reference first). Compare figures are not per-run stems, so nothing is registered in ``_BUILDERS`` /
# ``AXIS_BUILDERS`` (verify criterion 11 would then demand them in every run).
from teb_vae.classifier.metrics import cohort_digest, run_names  # noqa: E402

COMPARE_ROC = "compare_roc"
COMPARE_METRIC_TYPES = "compare_metric_types_{axis}"
COMPARE_FOREST = "compare_forest"
COMPARISON_MD = "comparison.md"
#: The ``roc_points`` columns the Q1 ROC panels read.
Q_ROC_COLS = ["level", "fold", "variant", "fpr", "tpr", "tpr_lo", "tpr_hi", "n_pos", "n_neg"]


def _compare_run(run: Path, name: str) -> Dict[str, Any]:
    """One compared run for Q1/Q3: ``name``, ``path``, classifier config ``c``, primary model ``pm``, the summary
    ``head``line, and its primary model's pooled test rows of ``roc_points`` (``roc``) and of ``metrics`` (``met``:
    :data:`TR_COLS` of the P2 and bin rows). Frames keep their columns when the run has no evaluation (empty panels)."""
    from teb_vae.classifier.config import load

    ids = pd.read_parquet(run / "predictions" / "guids.parquet", columns=["model_id", "seed"]).drop_duplicates()
    pm = primary_model(sorted(ids.itertuples(index=False, name=None)))
    tb = run / "evaluation" / "tables"
    rows = [("model_id", "==", pm[0]), ("seed", "==", pm[1]), ("split", "==", "test"), ("fold", "==", "pooled")]
    roc, met = tb / "roc_points.parquet", tb / "metrics.parquet"
    res = (_read_json(run / "evaluation" / "summary.json") or {}).get("results") or {}
    return {"name": name, "path": run, "c": classifier_cfg(load(run / "config.resolved.yaml")), "pm": pm,
            "head": res.get("headline") or {},
            "roc": (pd.read_parquet(roc, filters=rows) if roc.is_file() else pd.DataFrame()).reindex(
                columns=Q_ROC_COLS),
            "met": (pd.read_parquet(met, columns=TR_COLS, filters=[*rows, ("point", "in", ["n/a", "bin"])])
                    if met.is_file() else pd.DataFrame()).reindex(columns=TR_COLS)}


def _compare_roc(R: List[Dict[str, Any]], c: Dict[str, Any]) -> Any:
    """Q1: the R1 pooled GUID final-score ROC with its 95% cluster-bootstrap band, then the R2 committed-cumulative ROC
    at every checkpoint and at end; one colour per run (its primary model), AUC in the legends."""
    fs = _seam()
    pal, ats = fs.LINE_PALETTE, [None, *(f"{h:g}" for h in c["eval"]["checkpoints_h"]), "end"]
    fig, axes = _panels(len(ats), 4, 2.3)
    for ax, at in zip(axes, ats):
        stats = []
        for i, r in enumerate(R):
            color, roc = pal[i % len(pal)], r["roc"]
            if at is None:
                g, b = (_sel(roc, level="guid", fold="pooled", variant=v) for v in ("score_final", "score_final:band"))
                if len(b):
                    ax.fill_between(b["fpr"], b["tpr_lo"], b["tpr_hi"], color=color, alpha=0.12, lw=0)
            else:
                g = _sel(roc, level="online", fold="pooled", variant=f"committed_cumulative@{at}")
            if len(g):
                auc = np.trapezoid(g["tpr"].to_numpy(float), g["fpr"].to_numpy(float))
                ax.plot(g["fpr"], g["tpr"], color=color, lw=fs.LINE_EMPHASIS * 2, label=r["name"], zorder=3)
                stats.append((color, f"{auc:.2f}"))
        ax.set_title("Final score" if at is None else "All segments" if at == "end" else f"{at} h before delivery")
        if not ax.lines:
            _empty(ax)
            ax.set(xlim=(0, 1), ylim=(0, 1))
            continue
        _key(ax)
        _unit_square(ax)
        _stat_list(ax, stats)
        ax.set(xlabel="FPR", ylabel="sensitivity")
        fs.style_axes(ax)
    _tag(fig, "Run comparison · ROC")
    fs.caveat_note(fig, text="Checkpoint panels: committed cumulative ROC. Numbers: AUC of each run, in its colour.")
    return fig


def _compare_metric_types(R: List[Dict[str, Any]], c: Dict[str, Any], *, axis: str, pid: str) -> Any:
    """Q1: M1 of every run's primary model under ``pid``: three stacked rows (instantaneous, committed cumulative,
    committed overall) of sensitivity (solid, ○) and FPR (dashed, △) with their 95% bands on the ``eval.bin_h`` grid of
    ``axis``, one colour per run; the α and basis-time lines; the reference run's n strip; pooled test."""
    fs, pol = _seam(), _policy(c, pid)
    pal, d = fs.LINE_PALETTE, [_sel(r["met"], level="online", axis=axis, point="bin") for r in R]
    d = [x[x["subgroup"].isna()] for x in d]
    if not any(len(x) for x in d):
        return _all_empty()
    from matplotlib.lines import Line2D

    fig, grid, strips = _facets(1, len(TYPES), strip=(0,), row_h=2.0)
    rows = grid[0]
    _xlim(fig.axes, pd.concat([x["t"] for x in d]), axis, pol)
    for ax, mt in zip(rows, TYPES):
        for i, (r, x) in enumerate(zip(R, d)):
            rates = _rates(_sel(x, metric_type=mt), pid)
            for met, marker, ls, what in (("sens", "o", "-", "sens"), ("fpr", "^", "-", "FPR")):
                _series(ax, rates[met], c, color=pal[i % len(pal)], marker=marker, ls=ls, label=None)
        if ax.lines:
            if pol.get("alpha"):
                ax.axhline(pol["alpha"], ls=(0, (3, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE,
                           label=f"α = {pol['alpha']:g}")
            _unit_y(ax)
            fs.style_axes(ax)
        else:
            _empty(ax)
        _time_lines(ax, axis, pol)
        ax.set_title(TYPE_LABEL[mt].replace(chr(10), " "))
    rows[0].set_ylabel("rate")
    fs.add_key(fig, [Line2D([], [], color=pal[i % len(pal)], lw=2) for i in range(len(R))]
               + [Line2D([], [], color=fs.MUTED, marker="o", mec=fs.EDGE, mew=fs.MARKER_EDGE),
                  Line2D([], [], color=fs.MUTED, marker="^", mec=fs.EDGE, mew=fs.MARKER_EDGE)],
               [*(r["name"] for r in R), "sensitivity", "FPR"])
    _key(rows[0])
    _n_strip(strips[0], _sel(d[0], metric_type="instantaneous", metric="underpowered"), c)
    _time_lines(strips[0], axis, pol)
    for j in range(len(TYPES)):
        _x_time(_bottom(grid, strips, j), axis)
    _tag(fig, f"Run comparison · rates over time · policy {pid}")
    fs.caveat_note(fig, text=f"{UNDER_BIN.format(n=c['eval']['min_bin_class_n'])}. n strip: {R[0]['name']}.")
    return fig


def _compare_forest(Q: pd.DataFrame, c: Dict[str, Any], names: List[str], *, pid: str) -> Any:
    """Q2: each run minus the reference, ΔAUROC and Δsensitivity / Δspecificity at ``pid``, on all pooled test GUIDs and
    per subgroup member (families in the reference's order, members worst first). Paired-bootstrap 95% CIs, one colour
    per run. Filled: Holm-adjusted p < 0.05 within the family. Hollow: not significant, or underpowered (drawn at the
    estimate, without a CI). The DeLong and Nadeau-Bengio p of the whole population are in ``comparison.md``."""
    fs = _seam()
    cols = [("delta_auroc", None, "ΔAUROC"), ("delta_sens", pid, f"Δsensitivity ({pid})"),
            ("delta_spec", pid, f"Δspecificity ({pid})")]
    if Q.empty:
        return _all_empty(1, len(cols))
    q = Q.assign(subgroup=Q["subgroup"].fillna("overall").astype(str),
                 subgroup_value=Q["subgroup_value"].fillna("all GUIDs").astype(str), metric=Q["metric"].astype(str))
    fams = ["overall", *(f for f in subgroup_families(c) if f in set(q["subgroup"]))]
    cells = [(f, m) for f in fams for m in member_order(q.loc[q["subgroup"] == f, "subgroup_value"].unique())]
    pos, y = {}, 0.0
    for i, cell in enumerate(cells):
        y += 1.0 + (0.6 if i and cells[i - 1][0] != cell[0] else 0.0)
        pos[cell] = y
    pal, k = fs.LINE_PALETTE, max(len(names) - 1, 1)
    fig, axes = _figure(1, len(cols), max(2.6, 0.12 * y + 1.0), width=2.3 * len(cols) + 2.4)
    for ax, (met, p, title) in zip(axes[0], cols):
        g, drawn = q[(q["metric"] == met) & ((q["policy_id"] == p) if p else q["policy_id"].isna())], False
        for j, run in enumerate(names[1:], start=1):
            h = g[g["run"] == run]
            under = h["underpowered"].astype(bool).to_numpy()
            x = np.where(under, h["value_raw"].astype(float), h["value"].astype(float))
            dy = 0.5 * ((j - 1) / k - 0.5 + 0.5 / k)  # runs side by side within a member's row
            ys = np.array([pos[(f, m)] + dy for f, m in zip(h["subgroup"], h["subgroup_value"])])
            color, filled = pal[j % len(pal)], (h["p_holm"].astype(float) < 0.05).to_numpy()
            ax.hlines(ys, h["ci_lo"].astype(float), h["ci_hi"].astype(float), colors=color, lw=fs.LINE_REGULAR * 1.3)
            ax.scatter(x, ys, s=18, edgecolors=np.where(filled, fs.EDGE, color),
                       facecolors=np.where(filled, color, "white"), linewidths=0.9, zorder=3,
                       label=f"{run} − {names[0]}")
            drawn |= bool(np.isfinite(x).any())
        ax.axvline(0.0, ls=(0, (2, 2)), color=fs.MUTED, lw=fs.LINE_HAIRLINE)
        ax.set(title=title, ylim=(y + 1.0, 0.0))
        ax.set_yticks(list(pos.values()), [f"{f}: {m}" if ax is axes[0, 0] else "" for f, m in pos],
                      fontsize=fs.FONT_SMALL)
        ax.tick_params(axis="y", length=0)
        fs.style_axes(ax, grid="x")
        if not drawn:
            _empty(ax)
    _key(axes[0, 0])
    _tag(fig, "Run comparison · differences to the reference")
    fs.caveat_note(fig, text=f"Filled: Holm p < 0.05. Hollow: not significant or n < {c['eval']['min_subgroup_n']} per class.")
    return fig


def comparison_md(R: List[Dict[str, Any]], Q: pd.DataFrame, out: Path, *, pid: str, digest: str) -> Path:
    """Q3 ``comparison.md``. It lists the runs (the reference first), their shared cohort digest and the bootstrap.
    The ablation table has one row per run (its primary model): only the config leaves that differ across the runs
    (``verify.config_arms``), the headline (GUID AUROC fold mean ± SD, the primary, and pooled test with CI; pAUC;
    sensitivity at ``pid`` pooled with CI and fold mean ± SD; FPR at ``pid``) and the Q2 Δ vs the reference on all
    GUIDs (ΔAUROC with its bootstrap, DeLong and Nadeau-Bengio p). Then the Q2 whole-population rows at every compared
    policy."""
    from teb_vae.classifier.verify import _ABSENT, _render, _table, config_arms

    ev, alpha = R[0]["c"]["eval"], primary_alpha(R[0]["c"]["eval"])
    arms, leaves = config_arms([r["c"] for r in R])
    overall = Q[Q["subgroup"].isna()] if len(Q) else Q

    def h(r: Dict[str, Any], p: str, met: str) -> Dict[str, Any]:
        return r["head"].get(f"{r['pm'][0]}|{r['pm'][1]}|guid|{p}|{met}") or {}

    def sd(e: Dict[str, Any]) -> str:
        return f"{_cell(e.get('fold_mean'))} ± {_cell(e.get('fold_sd'))}"

    def delta(r: Dict[str, Any], met: str, p: Optional[str]) -> str:
        x = _sel(overall, run=r["name"], metric=met)
        x = x[(x["policy_id"] == p) if p else x["policy_id"].isna()] if len(x) else x
        tests = "".join(f", {name} p {_cell(float(x[col].iloc[0]))}" for col, name in (("p_delong", "DeLong"),
                                                                                       ("p_nb", "NB")) if col in x)
        return "-" if not len(x) else (_ci(x["value"].iloc[0], [x["ci_lo"].iloc[0], x["ci_hi"].iloc[0]])
                                       + f", p {_cell(x['p_value'].iloc[0])}" + (tests if met == "delta_auroc" else ""))

    header = ["run", *arms, "model", "AUROC fold mean ± SD", "AUROC pooled test [CI]", f"pAUC@{alpha:g} fold mean ± SD",
              f"sens ({pid}) pooled test [CI]", f"sens ({pid}) fold mean ± SD", f"FPR ({pid}) pooled test [CI]",
              "ΔAUROC [CI], p (bootstrap, DeLong, NB)", f"Δsens ({pid}) [CI], p"]
    rows = []
    for i, (r, lv) in enumerate(zip(R, leaves)):
        au, sens, fpr = h(r, "threshold_free", "auroc"), h(r, pid, "sens"), h(r, pid, "fpr")
        rows.append([r["name"] + (" (reference)" if i == 0 else ""), *(_render(lv.get(k, _ABSENT)) for k in arms),
                     f"{r['pm'][0]} (seed {r['pm'][1]})", sd(au), _ci(au.get("test"), au.get("test_ci")),
                     sd(h(r, "threshold_free", f"pauc@{alpha:g}")), _ci(sens.get("test"), sens.get("test_ci")),
                     sd(sens), _ci(fpr.get("test"), fpr.get("test_ci")),
                     *(("reference",) * 2 if i == 0 else (delta(r, "delta_auroc", None), delta(r, "delta_sens", pid)))])
    per = [[str(x.run), str(x.policy_id) if isinstance(x.policy_id, str) else "threshold-free", x.metric,
            _cell(x.reference_value), _cell(x.run_value), _ci(x.value, [x.ci_lo, x.ci_hi]), _cell(x.p_value),
            _cell(float(getattr(x, "p_delong", np.nan))), _cell(float(getattr(x, "p_nb", np.nan))),
            str(int(x.n_pos + x.n_neg))] for x in overall.itertuples()] if len(overall) else []
    b, runs = ev["bootstrap"], ", ".join(f"`{r['path']}`" for r in R)
    L = ["# Model comparison (Q1-Q3)", "",
         f"Runs (the first is the reference): {runs}.", "",
         f"Shared cohort digest: `{digest}`. Each run contributes its primary model. Δ = run − reference on the same "
         f"pooled OOF test GUIDs (shared test GUIDs deduplicated per data.shared_test_policy). CIs: paired "
         f"outcome-stratified patient-cluster percentile bootstrap, B = {b['resamples']}, seed {b['seed']}, the same "
         f"draws for both runs; p: two-sided bootstrap. ΔAUROC also carries the DeLong p (Sun & Xu 2014, on the pooled "
         f"GUID scores, GUIDs independent) and the Nadeau-Bengio corrected resampled t-test p on the per-fold test "
         f"AUROCs (variance factor 1/k + 1/(k-1), df k-1). Per-subgroup rows (Holm within the family) are in "
         f"`comparison.parquet` and `figures/{COMPARE_FOREST}`.", "",
         "## Ablation table", "",
         f"Config columns: only the keys whose values differ across the runs ({len(arms)}). AUROC: the per-fold mean ± "
         "SD is primary, the pooled OOF AUROC secondary (§11.7).", "",
         *_table(header, rows), "",
         "## Paired Δ vs the reference, all GUIDs", "",
         *(_table(["run", "policy", "metric", "reference", "run", "Δ [95% CI]", "p", "DeLong p", "NB p", "N"], per)
           if per
           else ["_no data_"]),
         "", "Naive cross-validation confidence intervals under-cover across folds (Bates 2024).", ""]
    path = out / COMPARISON_MD
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def _render_compare(build: Callable[[], Any], path: Path) -> None:
    _render(build(), path)


def compare_report(run_dirs: Any, out: Any, *, policy: Optional[str] = None) -> int:
    """The Q1 figures, the Q2 forest and Q3 ``comparison.md`` of a ``compare`` into ``out``. Figures go to
    ``figures/<stem>`` in the reference's ``eval.figure_formats``, drawn from each run's evaluation tables and
    ``out/comparison.parquet``, under ``policy`` (default: the reference's primary policy). Each figure and the markdown
    are fail-soft steps. Returns 1 if any raised, else 0."""
    from teb_vae.lag_attn.eval.report import Report

    paths, out = [Path(r) for r in run_dirs], Path(out)
    R = [_compare_run(p, n) for p, n in zip(paths, run_names(paths))]
    c, names, Q = R[0]["c"], [r["name"] for r in R], pd.read_parquet(out / "comparison.parquet")
    pid = policy or c["eval"]["primary_policy"]
    figs = {COMPARE_ROC: partial(_compare_roc, R, c), COMPARE_FOREST: partial(_compare_forest, Q, c, names, pid=pid),
            **{COMPARE_METRIC_TYPES.format(axis=a): partial(_compare_metric_types, R, c, axis=a, pid=pid)
               for a in c["eval"]["time_axes"]}}
    rep, fmts = Report(), c["eval"]["figure_formats"]
    _seam().configure_figure_style(fmts[0])
    for fmt in fmts:
        _seam().figures.set_figure_format(fmt)
        for stem, build in figs.items():
            rep.step(f"{stem}.{fmt}", _render_compare, build, out / "figures" / stem)
    rep.step(COMPARISON_MD, comparison_md, R, Q, out, pid=pid, digest=cohort_digest(paths[0]))
    for record in rep.failed_steps:
        logger.error(f"compare step {record.name} failed:\n{record.traceback}")
    return rep.exit_code()
