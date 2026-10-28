r"""Figures and checks for one classification build: the screening, the selection and the folds.

The build runs in three steps. ``create_new_pipeline.py`` (``selection_only=True``) writes the
screening CSV and the fold pickle, ``build_minimization_folds.py`` re-folds the pool, and
``build_fold_datasets.py`` writes the HDF5 files. This module reads the outputs of the first two
from one folder and writes two sets of figures into ``<folder>/figures/``:

* ``overview/``: the dataset itself. Where every screened GUID went (classification pool,
  pretraining, too little signal), the healthy subgroup shares in each partition against the
  eligible population, the class balance per fold, signal hours, TLO presence, covariates,
  missingness and hospital mix.
* ``balance/``: how well the minimization folds balance each subgroup's covariates, written by
  ``plot_fold_balance.py`` (same folder as this file).

It also writes ``figures/summary.md``: the counts and a list of integrity checks, each marked
PASS or FAIL.

**Pretraining set.** It is computed with the rule of ``build_fold_datasets.py`` at its default
``pretrain_eligibility='eligible_3h'``: every healthy GUID with ``eligible_3h`` and no error that
is not a classification GUID.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401 -- matplotlib.ticker.PercentFormatter
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# plot_fold_balance puts the repository root on sys.path, sets the rcParams and imports the
# launch helper; this module reuses all three, with its palette and its balance pages.
import plot_fold_balance as pfb  # noqa: E402
from plot_fold_balance import (  # noqa: E402
    AXIS, CLASS_COLORS, CLASS_OF, GRID, INK, INK_2, MUTED, SUBGROUP_ORDER, SURFACE,
    missing_required, resolve_launch_args,
)

HEALTHY = ("healthy_bg_no_cs", "healthy_bg_cs", "healthy_no_bg_no_cs", "healthy_no_bg_cs")
UNHEALTHY = ("hie_no_cs", "hie_cs", "acidosis_no_cs", "acidosis_cs")
PARTITIONS = ("train", "val", "test")

#: Categorical slots of the reference palette, in their fixed order.
SLOT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
#: Where a screened GUID went. Gray steps for the GUIDs the build does not use.
FATE_COLORS = {
    "classification": SLOT[0], "pretraining": SLOT[2],
    "eligible, unused": SLOT[1], "too little signal": "#c3c2b7", "unreadable": MUTED,
}
PARTITION_COLORS = {"population": MUTED, "train": SLOT[0], "val": SLOT[2], "test": SLOT[1]}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f0f5fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

#: 0/1 covariates of ``cohort_covariates.csv`` shown as a share of 1s.
BINARY_COVARIATES = ("nullparity", "caesarean_prior", "gestational_diabetes",
                     "gestational_hypertension", "preeclampsia", "chronic_hypertension",
                     "chorioamnionitis", "intrapartum_fever")


# =============================================================================
# Inputs
# =============================================================================
SUMMARY_JSON = "classification_guid_selection_summary.json"


def resolve_healthy_eligibility(folder: Path, requested: Optional[str]) -> str:
    """Return the healthy eligibility rule of the build.

    Args:
        folder: The build folder.
        requested: ``'eligible_2h'``, ``'eligible_3h'`` or ``None``. ``None`` reads
            ``stats.healthy_eligibility`` from ``classification_guid_selection_summary.json``
            when the folder holds it, else falls back to ``'eligible_3h'``, the rule of builds
            made before the option existed.

    Returns:
        The rule.
    """
    if requested:
        return requested
    summary = folder / SUMMARY_JSON
    if summary.exists():
        with open(summary, encoding="utf-8") as f:
            rule = json.load(f).get("stats", {}).get("healthy_eligibility")
        if rule:
            return rule
    return "eligible_3h"


def load_inputs(folder: Path, healthy_eligibility: str) -> Dict[str, Any]:
    """Read the screening CSV, the pickle and the fold folder files.

    Args:
        folder: The folder with ``guid_screening_results.csv``,
            ``classification_dataset_records.pickle`` and the outputs of
            ``build_minimization_folds.py``.
        healthy_eligibility: See :func:`assign_fate`.

    Returns:
        ``screening`` (with a ``fate`` column), ``pickle`` and every table of
        :func:`plot_fold_balance.load_fold_folder`.
    """
    data = pfb.load_fold_folder(folder)
    screening = pd.read_csv(folder / "guid_screening_results.csv", dtype={"guid": str})
    with open(folder / "classification_dataset_records.pickle", "rb") as f:
        data["pickle"] = pickle.load(f)
    data["screening"] = assign_fate(screening, data["partitions"], healthy_eligibility)
    data["healthy_eligibility"] = healthy_eligibility
    return data


def assign_fate(screening: pd.DataFrame, partitions: pd.DataFrame,
                healthy_eligibility: str = "eligible_3h") -> pd.DataFrame:
    """Give every screened GUID the place the build puts it.

    The rules are those of ``select_classification_guids``: classification eligibility is
    ``eligible_2h`` for the unhealthy subgroups and *healthy_eligibility* for the healthy ones;
    pretraining takes the healthy GUIDs with ``eligible_3h`` outside classification.

    Args:
        screening: ``guid_screening_results.csv``.
        partitions: ``fold_partitions.csv``.
        healthy_eligibility: ``'eligible_3h'`` or ``'eligible_2h'``.

    Returns:
        ``screening`` with ``eligible`` and ``fate``: ``classification``, ``pretraining``,
        ``eligible, unused`` (eligible for classification, not chosen, and not a pretraining
        GUID), ``too little signal`` or ``unreadable``.
    """
    df = screening.copy()
    healthy = df["subgroup"].isin(HEALTHY)
    df["eligible"] = (np.where(healthy, df[healthy_eligibility], df["eligible_2h"])
                      & ~df["error"])
    in_cls = df["guid"].isin(set(partitions["tracing_guid"]))
    pretrain = healthy & df["eligible_3h"] & ~df["error"]
    df["fate"] = np.select(
        [df["error"], in_cls, pretrain, df["eligible"]],
        ["unreadable", "classification", "pretraining", "eligible, unused"],
        default="too little signal")
    return df


def _hours(rule: str) -> str:
    return "2 h" if rule == "eligible_2h" else "3 h"


def _ordered(present: Sequence[str]) -> List[str]:
    return [sg for sg in SUBGROUP_ORDER if sg in set(present)]


def _style(ax: plt.Axes, grid_axis: Optional[str] = "y") -> None:
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _title(fig: Figure, title: str, subtitle: str = "") -> None:
    fig.suptitle(title, x=0.02, ha="left")
    if subtitle:
        fig.text(0.02, 0.925, subtitle, color=INK_2, ha="left")


# =============================================================================
# Tables
# =============================================================================
def fate_table(screening: pd.DataFrame) -> pd.DataFrame:
    """GUIDs per subgroup and fate, with the totals and the classification fraction."""
    table = (screening.groupby(["subgroup", "fate"]).size().unstack(fill_value=0)
             .reindex(index=_ordered(screening["subgroup"]),
                      columns=list(FATE_COLORS), fill_value=0))
    table["screened"] = table[list(FATE_COLORS)].sum(axis=1)
    table["eligible"] = screening.groupby("subgroup")["eligible"].sum()
    table["classification_fraction_of_eligible"] = (table["classification"]
                                                    / table["eligible"].replace(0, np.nan))
    return table


def partition_counts(partitions: pd.DataFrame) -> pd.DataFrame:
    """GUIDs per fold, partition and subgroup."""
    return partitions.groupby(["fold", "partition", "subgroup"]).size().rename("n").reset_index()


def share_table(screening: pd.DataFrame, partitions: pd.DataFrame) -> pd.DataFrame:
    """Healthy subgroup shares: eligible population and each partition (mean, min, max).

    Returns:
        One row per (healthy subgroup, source) with ``mean``, ``min`` and ``max`` share of the
        healthy GUIDs; the population row has one value in all three.
    """
    eligible = screening[screening["eligible"] & screening["subgroup"].isin(HEALTHY)]
    pop = eligible["subgroup"].value_counts(normalize=True)
    rows = [{"subgroup": sg, "source": "population", "mean": pop.get(sg, 0.0),
             "min": pop.get(sg, 0.0), "max": pop.get(sg, 0.0)} for sg in HEALTHY]
    hp = partitions[partitions["subgroup"].isin(HEALTHY)]
    for part in PARTITIONS:
        sub = hp[hp["partition"] == part]
        if sub.empty:
            continue
        per_fold = (sub.groupby(["fold", "subgroup"]).size().unstack(fill_value=0)
                    .reindex(columns=list(HEALTHY), fill_value=0))
        share = per_fold.div(per_fold.sum(axis=1), axis=0)
        rows += [{"subgroup": sg, "source": part, "mean": share[sg].mean(),
                  "min": share[sg].min(), "max": share[sg].max()} for sg in HEALTHY]
    return pd.DataFrame(rows)


def prevalence_table(partitions: pd.DataFrame) -> pd.DataFrame:
    """Per fold and partition: unhealthy and healthy GUIDs, prevalence, healthy : unhealthy."""
    df = partitions.assign(unhealthy=partitions["subgroup"].isin(UNHEALTHY))
    out = (df.groupby(["fold", "partition"])["unhealthy"].agg(["sum", "size"])
           .rename(columns={"sum": "n_unhealthy", "size": "n_total"}).reset_index())
    out["n_healthy"] = out["n_total"] - out["n_unhealthy"]
    out["prevalence"] = out["n_unhealthy"] / out["n_total"]
    out["healthy_per_unhealthy"] = out["n_healthy"] / out["n_unhealthy"].replace(0, np.nan)
    return out


# =============================================================================
# Checks
# =============================================================================
def integrity_checks(data: Mapping[str, Any], n_folds: int) -> List[Tuple[str, bool, str]]:
    """Run the checks of ``summary.md``.

    Returns:
        ``(name, passed, detail)`` per check.
    """
    checks: List[Tuple[str, bool, str]] = []
    part, screening, cv = data["partitions"], data["screening"], data["pickle"]
    core = part[~part["extra_test"].astype(bool)]

    counts = pd.crosstab(core["tracing_guid"], core["partition"]).reindex(
        columns=list(PARTITIONS), fill_value=0)
    expected = {"test": 1, "val": 1, "train": n_folds - 2}
    bad = {p: int((counts[p] != n).sum()) for p, n in expected.items()}
    checks.append(("every pool GUID is in test once, val once and train K-2 times",
                   not any(bad.values()), f"GUIDs off the count per partition: {bad}"))

    dup = core.duplicated(["fold", "tracing_guid"]).sum()
    checks.append(("no GUID is in two partitions of one fold", dup == 0, f"{dup} repeats"))

    pickle_pool = {Path(p).stem for paths in cv["folds"]["fold_1"].values()
                   for sg_paths in paths.values() for p in sg_paths}
    extra = {Path(p).stem for paths in cv.get("test_augmentation", {}).values() for p in paths}
    fold_pool = set(core["tracing_guid"])
    checks.append(("the fold pool equals the pickle pool", pickle_pool - extra == fold_pool,
                   f"pickle {len(pickle_pool - extra)}, folds {len(fold_pool)}, "
                   f"only in pickle {len(pickle_pool - extra - fold_pool)}, "
                   f"only in folds {len(fold_pool - pickle_pool)}"))
    checks.append(("extra test GUIDs: pickle and fold folder agree",
                   extra == set(part.loc[part["extra_test"].astype(bool), "tracing_guid"]),
                   f"{len(extra)} extra test GUIDs"))

    cls = screening[screening["fate"] == "classification"]
    not_screened = fold_pool - set(screening["guid"])
    checks.append(("every classification GUID is in the screening CSV", not not_screened,
                   f"{len(not_screened)} missing"))
    inel = int((~cls["eligible"]).sum())
    hours = _hours(data.get("healthy_eligibility", "eligible_3h"))
    checks.append((f"every classification GUID is eligible (2 h unhealthy, {hours} healthy)",
                   inel == 0, f"{inel} ineligible"))

    sg_fold = part.drop_duplicates("tracing_guid").set_index("tracing_guid")["subgroup"]
    sg_scr = screening.set_index("guid")["subgroup"]
    common = sg_fold.index.intersection(sg_scr.index)
    wrong = int((sg_fold[common] != sg_scr[common]).sum())
    checks.append(("each GUID has the same subgroup in the folds and the screening",
                   wrong == 0, f"{wrong} differ"))

    unused = int(((screening["fate"] == "eligible, unused")
                  & screening["subgroup"].isin(UNHEALTHY)).sum())
    checks.append(("every eligible unhealthy GUID is in classification", unused == 0,
                   f"{unused} unused"))

    pre = set(screening.loc[screening["fate"] == "pretraining", "guid"])
    checks.append(("pretraining and classification are disjoint", not pre & fold_pool,
                   f"{len(pre)} pretraining GUIDs"))

    shares = share_table(screening, part)
    pop = shares[shares["source"] == "population"].set_index("subgroup")["mean"]
    test = shares[shares["source"] == "test"].set_index("subgroup")["mean"]
    gap = float((test - pop).abs().max()) if len(test) else float("nan")
    checks.append(("healthy test shares match the population (within 1 point)", gap <= 0.01,
                   f"largest gap {100 * gap:.2f} points"))
    return checks


# =============================================================================
# Figures
# =============================================================================
def fig_fates(table: pd.DataFrame, rule: str) -> Figure:
    """Where every screened GUID went, as 100 % bars per subgroup."""
    fates = [f for f in FATE_COLORS if table[f].sum() > 0]
    subgroups = list(table.index)
    shares = table[fates].div(table["screened"], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    left = np.zeros(len(subgroups))
    y = np.arange(len(subgroups))
    for fate in fates:
        ax.barh(y, shares[fate], left=left, height=0.64, color=FATE_COLORS[fate],
                edgecolor=SURFACE, linewidth=2, label=fate)
        for i, sg in enumerate(subgroups):
            n, w = int(table.loc[sg, fate]), shares.loc[sg, fate]
            if w >= 0.07:
                ax.text(left[i] + w / 2, i, f"{n:,}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if fate in ("classification", "unreadable") else INK)
        left += shares[fate].to_numpy()
    # The classification count is in the label too: in a large subgroup its segment is too
    # narrow to hold the number.
    ax.set_yticks(y, [f"{sg}  (n={int(table.loc[sg, 'screened']):,}; "
                      f"cls {int(table.loc[sg, 'classification']):,})" for sg in subgroups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of the screened GUIDs of the subgroup")
    _style(ax, "x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=len(fates))
    _title(fig, "Where every screened GUID goes",
           f"Eligible = at least 2 h (unhealthy) or {_hours(rule)} (healthy) of valid signal in "
           f"the last 6 h; pretraining keeps 3 h. Numbers are GUID counts.")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def fig_fraction(table: pd.DataFrame) -> Figure:
    """Classification GUIDs as a fraction of the eligible GUIDs, per subgroup."""
    subgroups = list(table.index)
    frac = table["classification_fraction_of_eligible"]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [CLASS_COLORS[CLASS_OF[sg]] for sg in subgroups]
    ax.bar(range(len(subgroups)), frac, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    for i, sg in enumerate(subgroups):
        ax.text(i, frac[sg], f"{100 * frac[sg]:.1f} %\n{int(table.loc[sg, 'classification']):,}"
                f" of {int(table.loc[sg, 'eligible']):,}", ha="center", va="bottom",
                fontsize=7, color=INK_2)
    ax.set_xticks(range(len(subgroups)), subgroups, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.18)
    _style(ax)
    _legend_classes(fig, subgroups)
    _title(fig, "Classification pool as a share of the eligible GUIDs",
           "Unhealthy subgroups use every eligible GUID; healthy subgroups use the selection "
           "fraction.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    return fig


def _legend_classes(fig: Figure, subgroups: Sequence[str]) -> None:
    classes = list(dict.fromkeys(CLASS_OF[sg] for sg in subgroups))
    fig.legend([Patch(color=CLASS_COLORS[c]) for c in classes], classes, loc="lower center",
               ncol=len(classes), bbox_to_anchor=(0.5, 0))


def fig_shares(shares: pd.DataFrame) -> Figure:
    """Healthy subgroup shares: population against train, val and test (fold mean, min-max)."""
    sources = [s for s in PARTITION_COLORS if s in set(shares["source"])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.8 / len(sources)
    for j, src in enumerate(sources):
        sub = shares[shares["source"] == src].set_index("subgroup").reindex(list(HEALTHY))
        x = np.arange(len(HEALTHY)) + (j - (len(sources) - 1) / 2) * width
        err = np.vstack([sub["mean"] - sub["min"], sub["max"] - sub["mean"]])
        ax.bar(x, sub["mean"], width=width * 0.92, color=PARTITION_COLORS[src],
               edgecolor=SURFACE, linewidth=1.5, label=src,
               yerr=err if src != "population" else None, ecolor=INK_2, capsize=2)
        for xi, v in zip(x, sub["mean"]):
            ax.text(xi, v + 0.012, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=6.5,
                    color=INK_2)
    ax.set_xticks(range(len(HEALTHY)), HEALTHY)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of the healthy GUIDs")
    _style(ax)
    ax.legend(loc="upper left")
    _title(fig, "Healthy subgroup mix: eligible population against each partition",
           "Bars are the mean over the folds; whiskers span the fold minimum to maximum. Labels "
           "are percent.")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


def fig_prevalence(prev: pd.DataFrame, n_folds: int) -> Figure:
    """Unhealthy prevalence and healthy : unhealthy per fold, two panels with one axis each."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    folds = np.arange(1, n_folds + 1)
    for ax, col, label, fmt in ((axes[0], "prevalence", "unhealthy prevalence", "pct"),
                                (axes[1], "healthy_per_unhealthy", "healthy : unhealthy", "x")):
        for part in PARTITIONS:
            sub = prev[prev["partition"] == part].set_index("fold").reindex(folds)
            ax.plot(folds, sub[col], color=PARTITION_COLORS[part], linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=part)
        # One end label: the three partitions sit within 0.2 points, so three labels collide.
        mean = prev[col].mean()
        ax.annotate(f"{100 * mean:.1f} %" if fmt == "pct" else f"{mean:.2f}",
                    (folds[-1], mean), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
        ax.set_xticks(folds)
        ax.set_xlabel("fold")
        ax.set_title(label)
        # From zero: an autoscaled axis turns a 0.1-point spread into a full-height swing.
        ax.set_ylim(0, 1.3 * prev[col].max())
        if fmt == "pct":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax)
    axes[0].legend(loc="best")
    _title(fig, "Class balance in every fold",
           "Unhealthy = HIE + acidosis. The classes are imbalanced by design; the loss handles "
           "it.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_partition_heatmaps(counts: pd.DataFrame, n_folds: int) -> Figure:
    """GUIDs per subgroup and fold, one annotated heatmap per partition."""
    subgroups = _ordered(counts["subgroup"])
    present = [p for p in PARTITIONS if p in set(counts["partition"])]
    fig, axes = plt.subplots(1, len(present), figsize=(14, 4.4), layout="constrained")
    for ax, part in zip(np.atleast_1d(axes), present):
        table = (counts[counts["partition"] == part].pivot(index="subgroup", columns="fold",
                                                           values="n")
                 .reindex(index=subgroups, columns=range(1, n_folds + 1)).fillna(0))
        # Each row scaled to its own mean, so a small subgroup's spread is as visible as a
        # large one's.
        rel = table.div(table.mean(axis=1).replace(0, np.nan), axis=0)
        image = ax.imshow(rel.to_numpy(dtype=float), cmap=SEQUENTIAL, vmin=0.8, vmax=1.2,
                          aspect="auto")
        for i in range(len(subgroups)):
            for j in range(n_folds):
                v = rel.iloc[i, j]
                ax.text(j, i, f"{int(table.iloc[i, j]):,}", ha="center", va="center",
                        fontsize=6.5, color=pfb._text_on(image.cmap(image.norm(v))))
        ax.set_xticks(range(n_folds), [str(f) for f in range(1, n_folds + 1)], fontsize=7)
        ax.set_yticks(range(len(subgroups)), subgroups if ax is np.atleast_1d(axes)[0] else [])
        ax.set_title(f"{part}  (total per fold ≈ {int(table.sum().mean()):,})")
        ax.set_xlabel("fold")
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=list(np.atleast_1d(axes)), shrink=0.85, pad=0.01)
    bar.set_label("count / subgroup mean over folds")
    bar.outline.set_visible(False)
    fig.suptitle("GUIDs per subgroup, fold and partition", x=0.02, ha="left")
    return fig


def _boxes(ax: plt.Axes, groups: List[np.ndarray], positions: Sequence[float], color: str,
           width: float) -> None:
    ax.boxplot(groups, positions=positions, widths=width, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=color, edgecolor=INK_2, linewidth=0.8, alpha=0.85),
               medianprops=dict(color=INK, linewidth=1.4),
               whiskerprops=dict(color=INK_2, linewidth=0.8),
               capprops=dict(color=INK_2, linewidth=0.8))


def fig_valid_hours(screening: pd.DataFrame, rule: str) -> Figure:
    """Valid signal hours in the last 6 h, per subgroup: classification against pretraining."""
    subgroups = _ordered(screening["subgroup"])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for k, fate in enumerate(("classification", "pretraining")):
        groups, pos = [], []
        for i, sg in enumerate(subgroups):
            v = screening.loc[(screening["subgroup"] == sg) & (screening["fate"] == fate),
                              "estimated_valid_hours_6h"].to_numpy()
            if len(v):
                groups.append(v)
                pos.append(i + (k - 0.5) * 0.36)
        if groups:
            _boxes(ax, groups, pos, FATE_COLORS[fate], 0.3)
    ax.set_xticks(range(len(subgroups)), subgroups, rotation=25, ha="right")
    ax.set_ylabel("estimated valid hours, last 6 h")
    _style(ax)
    ax.legend([Patch(color=FATE_COLORS[f]) for f in ("classification", "pretraining")],
              ["classification", "pretraining"], loc="lower right")
    _title(fig, "Valid signal in the last 6 h before delivery",
           "Boxes: quartiles, whiskers 1.5 IQR, no outliers drawn. Unhealthy GUIDs need 2 h, "
           f"healthy GUIDs {_hours(rule)} (pretraining 3 h).")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def tlo_table(screening: pd.DataFrame) -> pd.DataFrame:
    """Share of GUIDs with a known labour onset (TLO), per subgroup and group of GUIDs."""
    elig = screening[screening["eligible"]]
    out = pd.DataFrame({
        "eligible": elig.groupby("subgroup")["has_tlo"].mean(),
        "classification": elig[elig["fate"] == "classification"].groupby("subgroup")["has_tlo"]
        .mean(),
        "pretraining": elig[elig["fate"] == "pretraining"].groupby("subgroup")["has_tlo"].mean(),
    })
    return out.reindex(_ordered(out.index))


def fig_tlo(table: pd.DataFrame) -> Figure:
    """TLO presence per subgroup: eligible population, classification pool, pretraining."""
    colors = {"eligible": MUTED, "classification": FATE_COLORS["classification"],
              "pretraining": FATE_COLORS["pretraining"]}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.26
    for j, col in enumerate(colors):
        x = np.arange(len(table)) + (j - 1) * width
        vals = table[col].to_numpy()
        ax.bar(x, np.nan_to_num(vals), width=width * 0.92, color=colors[col], edgecolor=SURFACE,
               linewidth=1.5, label=col)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=6.5,
                        color=INK_2)
    ax.axhline(0.75, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.45, 0.735, "75 % = old no-BG TLO quota", ha="left", va="top",
            fontsize=7, color=INK_2, zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.set_xticks(range(len(table)), table.index, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share with a known labour onset")
    _style(ax)
    ax.legend(loc="upper left", ncol=3)
    _title(fig, "Known labour onset (TLO) per subgroup",
           "Builds with a no-BG fraction draw at random, so classification matches eligible; the "
           "old deficit rule drew no-BG at 75 % TLO.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_covariates(raw: pd.DataFrame) -> Figure:
    """Raw continuous covariates of the pool, per subgroup."""
    subgroups = _ordered(raw["subgroup"])
    columns = list(pfb.RAW_CONTINUOUS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, col in zip(axes.ravel(), columns):
        groups = [raw.loc[raw["subgroup"] == sg, col].dropna().to_numpy() for sg in subgroups]
        for i, (sg, g) in enumerate(zip(subgroups, groups)):
            if len(g):
                _boxes(ax, [g], [i], CLASS_COLORS[CLASS_OF[sg]], 0.6)
        ax.set_xticks(range(len(subgroups)), subgroups, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{col} ({pfb.RAW_CONTINUOUS[col][1]})")
        _style(ax)
    classes = list(dict.fromkeys(CLASS_OF[sg] for sg in subgroups))
    fig.legend([Patch(color=CLASS_COLORS[c]) for c in classes], classes, loc="outside lower "
               "center", ncol=len(classes))
    fig.suptitle("Pool covariates per subgroup (quartiles, whiskers 1.5 IQR, no outliers)",
                 x=0.02, ha="left")
    return fig


def _annotated_heatmap(table: pd.DataFrame, title: str, subtitle: str, label: str,
                       fmt: str = "{:.0f}", scale: float = 100.0) -> Figure:
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * table.shape[1] + 3), 0.45 * len(table) + 2.4))
    values = table.to_numpy(dtype=float)
    vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
    image = ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(scale * v), ha="center", va="center", fontsize=7,
                        color=pfb._text_on(image.cmap(image.norm(v))))
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.02)
    bar.set_label(label)
    bar.outline.set_visible(False)
    _title(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def binary_table(covariates: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Share of 1s of each 0/1 covariate per subgroup (missing values left out)."""
    df = assignment[["tracing_guid", "subgroup"]].merge(covariates, on="tracing_guid")
    cols = [c for c in BINARY_COVARIATES if c in df]
    out = df.groupby("subgroup")[cols].agg(lambda s: pd.to_numeric(s, errors="coerce").mean())
    return out.reindex(_ordered(out.index))


def missing_table(covariates: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Share of missing values of each covariate per subgroup."""
    df = assignment[["tracing_guid", "subgroup"]].merge(covariates, on="tracing_guid")
    flags = [c for c in df.columns if c.endswith("_missing")]
    out = df.groupby("subgroup")[flags].mean()
    out.columns = [c[: -len("_missing")] for c in flags]
    out = out.loc[:, out.max() > 0]
    return out.reindex(_ordered(out.index))


def hospital_table(covariates: pd.DataFrame, assignment: pd.DataFrame,
                   declared: Sequence[str]) -> pd.DataFrame:
    """Share of each hospital per subgroup, hospitals as ``Hnn`` codes."""
    df = assignment[["tracing_guid", "subgroup"]].merge(
        covariates[["tracing_guid", "hospital_guid"]], on="tracing_guid")
    df["hospital"] = df["hospital_guid"].fillna("_missing").map(
        lambda h: pfb.hospital_code(h, declared))
    out = pd.crosstab(df["subgroup"], df["hospital"], normalize="index")
    return out.reindex(index=_ordered(out.index), columns=sorted(out.columns))


# =============================================================================
# Summary
# =============================================================================
def _md(table: pd.DataFrame) -> str:
    """Render a table as GitHub Markdown, without the optional ``tabulate`` dependency."""
    def cell(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.2f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    header = [str(table.index.name or "")] + [str(c) for c in table.columns]
    # Row by row from itertuples: to_numpy() on mixed dtypes would turn every count into a float.
    rows = [[cell(row[0])] + [cell(v) for v in row[1:]] for row in table.itertuples()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def write_summary(path: Path, data: Mapping[str, Any], fates: pd.DataFrame,
                  shares: pd.DataFrame, prev: pd.DataFrame, tlo: pd.DataFrame,
                  checks: List[Tuple[str, bool, str]], balance_note: str) -> None:
    """Write ``summary.md``: counts, shares, class balance, TLO and the checks."""
    rc = data["run_config"]
    lines = ["# Classification build summary", "",
             f"Folds: {rc['n_folds']} ({rc['test_mode']} mode); pool {rc['n_pool_guids']:,} GUIDs; "
             f"extra test {rc['n_extra_test_guids']:,}; unstratified "
             f"{len(rc.get('unstratified_guids', []))}. Healthy eligibility: "
             f"{data['healthy_eligibility']} (unhealthy eligible_2h, pretraining eligible_3h).",
             "",
             "## Where every screened GUID goes", "",
             _md(fates.assign(classification_fraction_of_eligible=lambda t: (
                 100 * t["classification_fraction_of_eligible"]))
                 .rename(columns={"classification_fraction_of_eligible": "cls % of eligible"})),
             "", "## Healthy subgroup shares (percent of healthy GUIDs; fold mean)", ""]
    pivot = 100 * shares.pivot(index="subgroup", columns="source", values="mean")
    lines += [_md(pivot.reindex(index=list(HEALTHY))), "",
              "## Class balance (mean over folds; prevalence in percent)", ""]
    agg = prev.groupby("partition")[["n_unhealthy", "n_healthy", "prevalence",
                                      "healthy_per_unhealthy"]].mean()
    agg["prevalence"] = 100 * agg["prevalence"]
    lines += [_md(agg.reindex(list(PARTITIONS))), "",
              "## Known labour onset (percent)", "", _md(100 * tlo), "",
              "## Checks", ""]
    lines += [f"- **{'PASS' if ok else 'FAIL'}** {name}: {detail}" for name, ok, detail in checks]
    lines += ["", "## Fold balance", "", balance_note, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Entry point
# =============================================================================
def main(folder: str, out_dir: Optional[str] = None, n_random: int = 2000, seed: int = 0,
         skip_balance: bool = False, healthy_eligibility: Optional[str] = None) -> int:
    """Write the overview figures, the balance report and ``summary.md``.

    Args:
        folder: The folder with the screening CSV, the pickle and the fold folder files.
        out_dir: Output directory; ``None`` writes ``<folder>/figures``.
        n_random: Random splits per (subgroup, axis) pair of the balance report.
        seed: Seed of the random splits.
        skip_balance: Skip the balance report (the slow part).
        healthy_eligibility: See :func:`resolve_healthy_eligibility`.

    Returns:
        The process exit code: 1 if a check fails.
    """
    folder_path = Path(folder)
    out = Path(out_dir) if out_dir else folder_path / "figures"
    overview = out / "overview"
    overview.mkdir(parents=True, exist_ok=True)
    rule = resolve_healthy_eligibility(folder_path, healthy_eligibility)
    print(f"healthy eligibility: {rule}")
    data = load_inputs(folder_path, rule)
    n_folds = int(data["run_config"]["n_folds"])
    screening, part = data["screening"], data["partitions"]

    fates = fate_table(screening)
    shares = share_table(screening, part)
    prev = prevalence_table(part)
    counts = partition_counts(part)
    tlo = tlo_table(screening)
    raw = pfb.raw_covariates(data["covariates"], data["assignment"])
    declared = data["metadata"]["columns"].get("hospital_guid", {}).get("levels") or []
    binary = binary_table(data["covariates"], data["assignment"])
    missing = missing_table(data["covariates"], data["assignment"])
    hospitals = hospital_table(data["covariates"], data["assignment"], declared)
    for name, table in (("fates", fates), ("healthy_shares", shares), ("class_balance", prev),
                        ("partition_counts", counts), ("tlo_share", tlo),
                        ("binary_covariates", binary), ("missing_share", missing),
                        ("hospital_share", hospitals)):
        table.to_csv(overview / f"{name}.csv")

    pages = [
        ("01_screening_fates", lambda: fig_fates(fates, rule)),
        ("02_classification_fraction", lambda: fig_fraction(fates)),
        ("03_healthy_shares", lambda: fig_shares(shares)),
        ("04_class_balance", lambda: fig_prevalence(prev, n_folds)),
        ("05_partition_counts", lambda: fig_partition_heatmaps(counts, n_folds)),
        ("06_valid_hours", lambda: fig_valid_hours(screening, rule)),
        ("07_tlo_share", lambda: fig_tlo(tlo)),
        ("08_covariates", lambda: fig_covariates(raw)),
        ("09_binary_covariates", lambda: _annotated_heatmap(
            binary, "Binary covariates per subgroup (pool)",
            "Percent of the GUIDs with the condition; missing values left out.", "share")),
        ("10_missingness", lambda: _annotated_heatmap(
            missing, "Missing covariate values per subgroup (pool)",
            "Percent of the GUIDs with a missing value; covariates never missing are left out.",
            "share missing")),
        ("11_hospital_mix", lambda: _annotated_heatmap(
            hospitals, "Hospital mix per subgroup (pool)",
            "Percent of the subgroup's GUIDs per hospital; codes in balance/hospital_codes.csv.",
            "share")),
    ]
    with PdfPages(overview / "overview.pdf") as pdf:
        for name, make in pages:
            fig = make()
            fig.savefig(overview / f"{name}.png", dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {len(pages)} overview pages to {overview}")

    balance_note = "Not written yet (skip_balance=True and no balance/balance_table.csv)."
    if not skip_balance:
        pfb.main(str(folder_path), out_dir=str(out / "balance"), n_random=n_random, seed=seed)
    # A skipped run still reports a balance table that an earlier run wrote.
    if (out / "balance" / "balance_table.csv").exists():
        table = pd.read_csv(out / "balance" / "balance_table.csv")
        usable = table[(table["status"] != "dropped") & (table["n_levels"] > 1)]
        weak = usable[usable["ratio_to_random"] > 1.0]
        balance_note = (
            f"Observed S / random-median S over {len(usable)} (subgroup, axis) pairs: median "
            f"{usable['ratio_to_random'].median():.2f}, max {usable['ratio_to_random'].max():.2f}"
            f" (1 = as good as random). Pairs worse than random: {len(weak)}. Axis status "
            f"other than ok: " + ", ".join(
                f"{r.subgroup}/{r.axis}={r.status}" for r in table.itertuples()
                if r.status != "ok") + ". Details: balance/balance_table.csv.")

    checks = integrity_checks(data, n_folds)
    write_summary(out / "summary.md", data, fates, shares, prev, tlo, checks, balance_note)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"Summary: {out / 'summary.md'}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser. No flag has a default, so :data:`RUN_ARGS` fills the gaps.

    Returns:
        The parser.
    """
    p = argparse.ArgumentParser(description="Figures and checks for one classification build.")
    p.add_argument("--folder", dest="folder")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--n-random", dest="n_random", type=int)
    p.add_argument("--seed", dest="seed", type=int)
    p.add_argument("--healthy-eligibility", dest="healthy_eligibility",
                   choices=("eligible_2h", "eligible_3h"))
    p.add_argument("--skip-balance", dest="skip_balance", action=argparse.BooleanOptionalAction,
                   default=None)
    return p


def _cli(argv: Optional[List[str]] = None) -> int:
    """Merge the command line over :data:`RUN_ARGS` and run.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv[1:]``.

    Returns:
        The process exit code.
    """
    values, _ = resolve_launch_args(build_parser(), RUN_ARGS, argv)
    refusal = missing_required(values, ("folder",))
    if refusal:
        raise SystemExit(refusal)
    return main(**{k: v for k, v in values.items() if v is not None})


#: **Edit the values below and press Run.** A flag on the command line overrides its entry.
RUN_ARGS: Dict[str, Any] = {
    # str -- folder with guid_screening_results.csv, classification_dataset_records.pickle and
    #        the build_minimization_folds.py outputs. Required.
    "folder": r"C:\Users\mahdi\Desktop\teb_vae_model\tmp\fold_counts",

    # str | None -- output directory. None = <folder>/figures.
    "out_dir": None,

    # int -- random splits per (subgroup, axis) pair of the balance report.
    "n_random": 2000,

    # int -- seed of the random splits.
    "seed": 0,

    # bool -- skip the balance report.
    "skip_balance": False,

    # "eligible_2h" | "eligible_3h" | None -- the healthy classification rule of the build.
    #        None = read it from classification_guid_selection_summary.json in the folder,
    #        else "eligible_3h".
    "healthy_eligibility": None,
}


if __name__ == "__main__":
    sys.exit(_cli())
