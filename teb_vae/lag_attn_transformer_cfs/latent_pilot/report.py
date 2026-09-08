r"""Figure and report generators.

Written in LP-13. Three figures and one written report, all of them assembled from measured records
and none of them holding a number of its own.

Figures, all rendered standalone to the run directory through
``teb_vae.lag_attn.eval.figures`` -- its style configuration, colour helpers and renderer -- so a
pilot figure looks like every other figure this repository produces and no external dashboard is
involved. Class colouring, captions and explained-variance labels are consistent across all three,
because all three read the same :data:`CAPTIONS` and the same
:func:`~teb_vae.lag_attn.eval.figures.group_colors` mapping the training plots use.

**Figure 1 -- the latent space before and after.** The single label-free training PCA from
``analyze.py``, applied to both models with **identical axes** on the two panels, showing final-hour
recording bags coloured healthy / acidosis / HIE with a deterministic subset of paired before/after
arrows. Explained variance is stated. The same map is reused for the trajectories, so the two figures
share one coordinate system.

**Figure 2 -- separation along the learned direction.** Test distributions of $w^\top S(v) + b$ before
and after, labelled explicitly as a **supervised** axis, with recording-level AUROC, average precision
and counts. Separation can be visible here while the leading principal components carry other large
sources of variation. The score is not an independently discovered biomarker, its sigmoid is not
calibrated severity, and the two models' logit scales are not comparable as distances -- a caption
states so rather than leaving the reader to infer it.

**Figure 3 -- evolution over the last three hours.** Group means with GUID-bootstrap bands over the
six bins, the recording count at the foot of each bin, the supervised final hour marked, and signed hours $-3$ to
$0$ so delivery sits at the right. A small prespecified, seeded sample of individual trajectories is
overlaid, with raw FHR/UP excerpts where available. Observed points and real gaps are marked; no line
is extended to delivery without data, and no trace is chosen for looking convincing --
:func:`select_traces` draws from the class strata with a seeded generator and never reads a score.

The report generator assembles the measured run: provenance (checkpoint identity and digest,
statistics file, software revision, fold and seed), cohort counts and coverage with every exclusion
reason, the paired held-out metrics with intervals, the controls, the preservation gate results, the
temporal and subgroup findings, which candidate selection actually chose -- including the case where
the frozen model was retained -- the stated limitations, and the exact commands to reproduce the run.

Two rules the generator enforces rather than trusts: a missing value is rendered as missing, and no
example number is ever emitted as if it had been measured. Every figure in the written report comes
out of the record :func:`build_report` was handed; a section whose record is absent says so, in the
words of :func:`_missing`, and a section whose record is present but whose interval could not be
estimated prints the reason the estimator gave. A report that separates a software failure from
insufficient cohort support from an inconclusive scientific finding is worth more than one that
reads uniformly successful.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from teb_vae.lag_attn.eval import labels
from teb_vae.lag_attn_transformer_cfs.latent_pilot import analyze, data
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import PilotConfigError

#: Subdirectory of the run every figure is written into.
FIGURE_DIRNAME = "figures"

#: On-disk name of the written report.
REPORT_FILENAME = "report.md"

#: Figure name stems. Extension-less on purpose: the format is the run's, decided once by
#: :func:`configure_figures`, and :func:`~teb_vae.lag_attn.eval.figures.render_figure` refuses a
#: stem that carries one.
FIGURE_LATENT_SPACE = "figure1_latent_space"
FIGURE_SUPERVISED_AXIS = "figure2_supervised_axis"
FIGURE_TRAJECTORIES = "figure3_trajectories"

#: The coverage control's sensitivity view. Not one of the three main figures: it exists so that a
#: separation can be read against how the recordings were ascertained and how late they were still
#: observed, which is the alternative explanation §7.3 requires to be looked at rather than argued
#: away.
FIGURE_COVERAGE_SPACE = "figure1b_latent_space_by_coverage"

#: The classification readout. Not part of the original three: those answer whether the latent
#: moved and where the cohort sits in it, while these answer how well the score separates the two
#: outcome groups -- overall on the final-hour bags, and separately in every trajectory bin. They
#: are drawn from tables the evaluation stage already wrote, so they can be produced for a finished
#: run without refitting anything.
FIGURE_ROC_PR = "figure4_roc_pr"
FIGURE_CONFUSION = "figure5_confusion"
FIGURE_METRICS_TIME = "figure6_metrics_vs_time"
FIGURE_ROC_BINS = "figure7_roc_by_time_bin"
FIGURE_COUNTS_TIME = "figure8_confusion_counts_vs_time"

#: The class axis, in the order every panel draws it, taken from the repository's own class table
#: rather than restated -- a class added there reaches these figures without an edit here.
PLOT_CLASSES: Tuple[str, ...] = tuple(labels.CLASS_NAMES.values())

#: What a value that was not measured is printed as. One string, so a reader can grep the report
#: for everything this run could not establish.
MISSING = "not measured"

#: How many individual trajectories Figure 3 overlays unless told otherwise. Small on purpose: the
#: point is to show that group means are means of real, gappy traces, not to redraw the cohort.
DEFAULT_N_TRACES = 6

#: How many before/after arrow pairs Figure 1 draws. A subset, because sixty arrows are a texture
#: rather than a comparison.
DEFAULT_N_ARROWS = 12

#: The module the reproduction commands name. Written out rather than imported from ``run.py``,
#: which imports the stage bodies and would therefore import this module back.
RUNNER_MODULE = "teb_vae.lag_attn_transformer_cfs.latent_pilot.run"

#: The caption each figure carries, and the same text the report prints beside it. One definition,
#: two consumers: a caption that disagreed with the report would be the reader's problem to resolve.
CAPTIONS: Dict[str, str] = {
    FIGURE_LATENT_SPACE: (
        "One label-free PCA, fitted on training recording-bin summaries of both model versions and "
        "applied unchanged to both panels, which therefore share axes. Arrows join the same "
        "recording before and after adaptation for a seeded subset. Position along these axes is "
        "not a severity scale."
    ),
    FIGURE_SUPERVISED_AXIS: (
        "The SUPERVISED axis: each model's own fitted direction w'S(v)+b on held-out recordings. "
        "Separation here is what the classification loss optimised, not an independently "
        "discovered biomarker; the sigmoid of this score is not calibrated risk, and the two "
        "models' logit scales are not comparable as distances."
    ),
    FIGURE_COVERAGE_SPACE: (
        "The map of Figure 1, coloured by how each recording was ascertained (blood gas, Caesarean) "
        "and by how late it was still observed. A cloud that separates mainly along these is not an "
        "outcome result. Colouring is descriptive: nothing here is adjusted for, and no stratum was "
        "fitted separately."
    ),
    # A format template, filled by :func:`figure_trajectories` from the run's own window settings.
    # ``CAPTIONS`` stays a plain ``Dict[str, str]``, which ``_figures_section`` and the logic tests
    # both index; what changes is that the sentence cannot state a geometry the config can move.
    FIGURE_TRAJECTORIES: (
        "Each model's FROZEN classifier, fitted on the final {supervised_hours:g} h, applied to "
        "every occupied {bin_hours:g} h bin. Bins outside the shaded supervised window are "
        "applications of that head outside the window it was fitted on. Counts at the foot of "
        "each bin are the recordings the band rests on; gaps are real absences and no line is "
        "extended to delivery without data. Raw FHR/UP excerpts are not produced by this "
        "pipeline."
    ),
    FIGURE_ROC_PR: (
        "Held-out ROC and precision-recall for the final-hour recording bags. The curves are "
        "threshold-free and therefore comparable between two models whose logit scales are not; "
        "the dots are the operating point of each model's threshold, which was chosen on "
        "validation and never here. PR chance is the adverse-outcome prevalence, not one half."
    ),
    FIGURE_CONFUSION: (
        "Confusion counts at each model's validation-selected threshold, on the same held-out "
        "recordings, with every rate derived from them. All panels share one colour scale, so a "
        "cell of the same colour is the same count. Counts are recordings, not segments or "
        "anchors, and a rate resting on a handful of them is not a smaller version of the same "
        "measurement."
    ),
    FIGURE_METRICS_TIME: (
        "Discrimination measured separately in each trajectory bin. The head is the frozen "
        "final-hour classifier and the threshold is the one selected on validation, so every bin "
        "outside the shaded supervised window is that rule applied where it was not fitted. The "
        "cohort is whoever was observed in a bin, so the adverse/total counts under each bin are "
        "part of the reading: a metric that moves towards delivery may be a trend or a change in "
        "who was still recorded. Bins carrying one class are gaps, not zeros."
    ),
    FIGURE_ROC_BINS: (
        "The ROC of every time bin that carried both outcome groups, nearest delivery first. A "
        "bin with no adverse recording has no panel rather than a diagonal that was never "
        "measured, so the panel count is itself a statement about coverage."
    ),
    FIGURE_COUNTS_TIME: (
        "What the rates of the previous figure are rates of: the four confusion cells per bin at "
        "each model's validation threshold. A sensitivity measured on two adverse recordings and "
        "one measured on thirty read identically as a rate and differently here."
    ),
}


def _figures() -> Any:
    """Import the repository's figure surface on first use.

    Deferred rather than top-level so that importing this module costs no matplotlib backend, and
    so the text half of the generator -- which is what the minimal logic checks exercise -- can be
    imported without a plotting stack at all. The same shape ``analyze.py`` uses for ``torch``.

    Returns:
        The :mod:`teb_vae.lag_attn.eval.figures` module.
    """
    from teb_vae.lag_attn.eval import figures

    return figures


def configure_figures(figure_format: Optional[str] = None) -> None:
    """Apply the repository's publication style and fix this run's figure format.

    Called once by the stage that draws, never at import: the style helper mutates global
    ``rcParams``, and doing that on import would restyle every other figure in the process.

    Args:
        figure_format: ``'pdf'``, ``'png'`` or any other format this matplotlib build writes.
            ``None`` keeps the active one, which is the figures module's default.
    """
    _figures().configure_figure_style(figure_format)


def _figure_dir(directory: Any) -> Path:
    """The run's figure subdirectory, created if absent.

    Args:
        directory: The run directory.

    Returns:
        ``<run>/figures``.
    """
    path = Path(directory) / FIGURE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Shared figure vocabulary
# =============================================================================
def class_palette(names: Sequence[str] = PLOT_CLASSES) -> Dict[str, str]:
    """The colour every panel gives each clinical class.

    Delegates to the figures module, which delegates to ``utils.style`` -- so a pilot figure of the
    healthy cohort is the same blue as the training figure of the same cohort.

    Args:
        names: The class labels appearing in a figure.

    Returns:
        Label to hex colour.
    """
    return _figures().group_colors([str(name) for name in names])


def variance_caption(projection: Any) -> str:
    """Name the share of weighted training variance each projection axis carries.

    Args:
        projection: The fitted :class:`~latent_pilot.analyze.Projection`.

    Returns:
        ``'PC1 41.2%, PC2 18.9% of weighted training variance'``, or a statement that the share is
        unavailable -- which happens when the weighted total variance was zero and the ratios are
        ``nan``. A percentage is never invented to fill the label.
    """
    ratios = np.asarray(projection.explained_variance_ratio, dtype=np.float64).reshape(-1)
    if ratios.size == 0 or not np.isfinite(ratios).any():
        return "explained variance unavailable"
    parts = [
        f"PC{index + 1} {100.0 * value:.1f}%" if np.isfinite(value) else f"PC{index + 1} {MISSING}"
        for index, value in enumerate(ratios.tolist())
    ]
    return f"{', '.join(parts)} of weighted training variance"


def axis_label(projection: Any, component: int) -> str:
    """The axis label of one projection component, carrying its own explained variance.

    Args:
        projection: The fitted projection.
        component: Zero-based component index.

    Returns:
        ``'PC1 (41.2%)'``, or ``'PC1'`` when the share is not finite.
    """
    ratios = np.asarray(projection.explained_variance_ratio, dtype=np.float64).reshape(-1)
    if component >= ratios.size or not np.isfinite(ratios[component]):
        return f"PC{component + 1}"
    return f"PC{component + 1} ({100.0 * float(ratios[component]):.1f}%)"


def score_scale_note(model: str, *, short: bool = False) -> str:
    """The disclosure that travels with every score axis of one classifier.

    Each model carries its own fitted $w$ and $b$, so its logits live on its own scale. A shift
    between two models' distributions is a difference of two fitted heads, and reading it as latent
    or physiological motion is the specific misreading this line exists to block.

    Args:
        model: The model's name.
        short: The panel-sized form, which fits on an axis label. The long form goes in the report
            and the figure caption, where there is room to say why.

    Returns:
        The disclosure.
    """
    if short:
        return f"{model}'s own logit scale -- not comparable across models"
    return (
        f"scores are {model}'s own fitted logits w'S(v)+b; the scale belongs to that head, so a "
        f"shift against another model's axis is a difference of two fitted heads and not latent or "
        f"physiological motion"
    )


def _palette(observed: Sequence[Any]) -> Dict[str, str]:
    """A colour for every label a panel actually draws.

    The known clinical classes keep the colour the rest of the repository gives them; anything else
    a panel groups by -- the binary outcome, a subgroup name -- gets a distinct palette entry rather
    than all falling back to one grey, which would draw two groups as the same line.

    Args:
        observed: The labels appearing in the panel.

    Returns:
        Label to hex colour, covering the clinical classes and every observed label.
    """
    extra = [str(value) for value in observed if str(value) not in PLOT_CLASSES]
    return class_palette(list(PLOT_CLASSES) + sorted(set(extra)))


def select_traces(
    frame: pd.DataFrame,
    *,
    n: int = DEFAULT_N_TRACES,
    seed: int = 0,
    group_column: str = labels.CLASS_COLUMN,
) -> List[str]:
    """Choose which individual recordings a figure draws, before anyone has seen their scores.

    **It never reads a score.** The selection is a seeded draw over the class strata of the GUID
    list, taken round-robin so a rare class is represented rather than sampled away, and identical
    for every model version and every rerun of the same seed. Picking the traces that look
    convincing is the failure this function exists to prevent, and the only defence that works is
    for the selector to be unable to see what it would be selecting on.

    Args:
        frame: Any recording-keyed table carrying :data:`~latent_pilot.data.GUID_COLUMN` and, where
            available, the class column. A score column may be present; it is ignored.
        n: How many recordings to draw.
        seed: Seeds the draw, so the figure is reproducible from the record alone.
        group_column: The stratifying column. Absent from the frame, every recording forms one
            stratum, which is a smaller claim than pretending the classes were balanced.

    Returns:
        The chosen GUIDs, in draw order. Fewer than ``n`` when the cohort holds fewer.
    """
    if data.GUID_COLUMN not in frame.columns:
        raise PilotConfigError(
            f"trace selection needs a {data.GUID_COLUMN!r} column; the table it was handed carries "
            f"{sorted(frame.columns)}."
        )
    if frame.empty or int(n) <= 0:
        return []
    strata: Dict[str, List[str]] = {}
    for _index, row in frame.iterrows():
        guid = str(row[data.GUID_COLUMN])
        key = str(row.get(group_column, "")) if group_column in frame.columns else ""
        members = strata.setdefault(key, [])
        if guid not in members:
            members.append(guid)

    generator = np.random.default_rng(int(seed))
    shuffled: Dict[str, List[str]] = {}
    for key in sorted(strata):
        members = sorted(strata[key])
        order = generator.permutation(len(members))
        shuffled[key] = [members[position] for position in order.tolist()]

    chosen: List[str] = []
    while len(chosen) < int(n) and any(shuffled.values()):
        for key in sorted(shuffled):
            if shuffled[key] and len(chosen) < int(n):
                chosen.append(shuffled[key].pop(0))
    return chosen


def with_class_names(frame: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    """Attach the outcome and clinical class of each recording to a summary table.

    The bag and window reductions are keyed by GUID and carry counts and times, not labels -- the
    reductions are label-blind by construction. The figures need the colour, so the join happens
    here, once, rather than inside three panels.

    Args:
        frame: A recording-keyed summary table.
        recordings: The recording table, carrying the outcome and the class name.

    Returns:
        A copy carrying :data:`~latent_pilot.data.OUTCOME_COLUMN` and
        :data:`~teb_vae.lag_attn.eval.labels.CLASS_COLUMN`. A GUID absent from ``recordings`` gets
        ``None`` in both rather than a guessed class.
    """
    outcomes = {
        str(row[data.GUID_COLUMN]): row.get(data.OUTCOME_COLUMN)
        for _index, row in recordings.iterrows()
    }
    names = {
        str(row[data.GUID_COLUMN]): row.get(labels.CLASS_COLUMN)
        for _index, row in recordings.iterrows()
    }
    out = frame.copy()
    guids = [str(value) for value in out[data.GUID_COLUMN].tolist()]
    out[data.OUTCOME_COLUMN] = [outcomes.get(guid) for guid in guids]
    out[labels.CLASS_COLUMN] = [names.get(guid) for guid in guids]
    return out


def _class_of(frame: pd.DataFrame) -> List[str]:
    """The class label of each row, falling back to the binary outcome's name."""
    if labels.CLASS_COLUMN in frame.columns:
        values = [
            str(value) if value is not None and not pd.isna(value) else ""
            for value in frame[labels.CLASS_COLUMN].tolist()
        ]
        if any(values):
            return values
    outcomes = frame.get(data.OUTCOME_COLUMN)
    if outcomes is None:
        return ["" for _ in range(len(frame))]
    return [
        "" if value is None or pd.isna(value) else ("adverse" if int(value) == 1 else "healthy")
        for value in outcomes.tolist()
    ]


# =============================================================================
# Figure 1 -- the latent space before and after, on one map
# =============================================================================
def _scatter_panel(
    ax: Any, points: np.ndarray, labels: Sequence[str], *,
    palette: Mapping[str, str], projection: Any, title: str,
) -> None:
    """Draw one projected cloud, one colour per label, on the shared map's axes.

    Shared by the main figure and the coverage view below, because the two answer the same question
    of the same points and differ only in what they colour by -- and a second copy of the scatter
    would be free to draw them at two sizes, two alphas or two axis labels.

    Args:
        ax: Target axes.
        points: ``(N, 2)`` projected coordinates.
        labels: One group label per row.
        palette: Label to colour.
        projection: The fitted map, for the axis labels and their explained variance.
        title: Panel title.
    """
    figures = _figures()
    values = [str(label) for label in labels]
    for label in sorted(set(values)):
        rows = [index for index, value in enumerate(values) if value == label]
        ax.scatter(
            points[rows, 0], points[rows, 1],
            s=18.0, alpha=0.8, linewidths=0.3,
            edgecolors=figures.COLOR_BLACK,
            color=palette.get(label, figures.COLOR_GRAY),
            label=f"{label or 'unlabelled'} (n={len(rows)})",
        )
    ax.set_title(title)
    ax.set_xlabel(axis_label(projection, 0))
    ax.set_ylabel(axis_label(projection, 1))
    ax.legend(loc="best", fontsize=6.0)
    figures.style_axes(ax)


def figure_latent_space(
    versions: Mapping[str, Tuple[pd.DataFrame, Any]],
    projection: Any,
    directory: Any,
    *,
    n_arrows: int = DEFAULT_N_ARROWS,
    seed: int = 0,
) -> Path:
    """Draw the final-hour bags of every model version on the one shared map.

    Both panels are drawn on the axes :func:`~latent_pilot.analyze.fit_projection` fitted once, on
    training data, without labels -- and on **identical limits**, computed over every point of every
    panel. Two panels with their own limits would show a movement that was a change of axis range.

    Args:
        versions: ``{name: (frame, standardized values)}``, one entry per model version, each frame
            carrying the GUID and the class name. The values are standardized by the frozen
            training scaler, which is the space the projection was fitted in.
        projection: The shared projection.
        directory: The run directory; the figure lands in its ``figures`` subdirectory.
        n_arrows: How many paired before/after arrows to draw on the second panel.
        seed: Seeds the arrow subset, which is chosen without reading any score or coordinate.

    Returns:
        The written path.

    Raises:
        PilotConfigError: If a version's values do not align with its frame.
    """
    figures = _figures()
    if np.asarray(projection.components).shape[0] < 2:
        raise PilotConfigError(
            f"this panel is two-dimensional and the projection carries "
            f"{np.asarray(projection.components).shape[0]} component(s); the protocol fits two."
        )
    names = list(versions)
    coordinates: Dict[str, np.ndarray] = {}
    for name in names:
        frame, values = versions[name]
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(frame):
            raise PilotConfigError(
                f"version {name!r} supplies {matrix.shape} values for {len(frame)} row(s); the "
                f"panel would colour one recording with another's class."
            )
        coordinates[name] = projection.transform(matrix)

    stacked = np.concatenate([values for values in coordinates.values()], axis=0) if names else (
        np.zeros((0, 2))
    )
    palette = _palette([
        label for name in names for label in _class_of(versions[name][0])
    ])
    fig, axes = figures.new_figure(1, max(len(names), 1), height_per_row=4.2, width=9.0)

    for column, name in enumerate(names):
        _scatter_panel(
            axes[0, column], coordinates[name], _class_of(versions[name][0]),
            palette=palette, projection=projection,
            title=f"{name}: final-hour recording bags",
        )

    # Paired arrows, on the last panel: the same recording before and after, for a seeded subset.
    if len(names) >= 2 and int(n_arrows) > 0:
        before_name, after_name = names[0], names[-1]
        before_frame, _ = versions[before_name]
        after_frame, _ = versions[after_name]
        before_rows = {
            str(guid): index
            for index, guid in enumerate(before_frame[data.GUID_COLUMN].tolist())
        }
        after_rows = {
            str(guid): index
            for index, guid in enumerate(after_frame[data.GUID_COLUMN].tolist())
        }
        shared = before_frame[
            before_frame[data.GUID_COLUMN].astype(str).isin(set(after_rows))
        ]
        drawn = 0
        for guid in select_traces(shared, n=int(n_arrows), seed=int(seed)):
            start = coordinates[before_name][before_rows[guid]]
            end = coordinates[after_name][after_rows[guid]]
            axes[0, len(names) - 1].annotate(
                "", xy=(end[0], end[1]), xytext=(start[0], start[1]),
                arrowprops={
                    "arrowstyle": "->", "linewidth": 0.7,
                    "color": figures.COLOR_GRAY, "alpha": 0.8,
                },
            )
            drawn += 1
        logger.info(
            f"figure 1: {drawn} paired arrow(s) from {before_name} to {after_name}, chosen by "
            f"seed {seed} without reading any coordinate or score"
        )

    if stacked.size and np.isfinite(stacked).any():
        finite = stacked[np.isfinite(stacked).all(axis=1)]
        pad = 0.05 * np.maximum(np.ptp(finite, axis=0), 1e-9)
        for column in range(len(names)):
            axes[0, column].set_xlim(finite[:, 0].min() - pad[0], finite[:, 0].max() + pad[0])
            axes[0, column].set_ylim(finite[:, 1].min() - pad[1], finite[:, 1].max() + pad[1])

    fig.suptitle(f"Latent space, shared map -- {variance_caption(projection)}")
    fig.text(0.01, 0.005, CAPTIONS[FIGURE_LATENT_SPACE], fontsize=6.0, wrap=True)
    return figures.render_figure(fig, _figure_dir(directory) / FIGURE_LATENT_SPACE)


def coverage_groupings(frame: pd.DataFrame) -> Dict[str, List[str]]:
    """Label each recording by how it was ascertained and by how late it was still observed.

    The two variables §7.3's coverage control asks the same picture to be coloured by. A cloud that
    separates mainly by blood-gas or Caesarean ascertainment, or mainly by how much signal survived
    to the end of the recording, is not an outcome result -- and the only way to see that is to
    colour the same map by those variables and look.

    Args:
        frame: One row per recording, carrying ``bg_label``, ``cs_label`` and
            ``last_anchor_hours`` where they exist.

    Returns:
        Grouping name -> one label per row. A variable the table does not carry, or one with too
        few distinct values to split, is returned as a single stated group rather than as an
        invented stratification.
    """
    strata: List[str] = []
    for _index, row in frame.iterrows():
        blood_gas = data._flag_is(row.get("bg_label"), True)
        caesarean = data._flag_is(row.get("cs_label"), True)
        strata.append(f"{'BG' if blood_gas else 'no BG'} / {'CS' if caesarean else 'no CS'}")

    hours = np.asarray(frame.get("last_anchor_hours", []), dtype=np.float64).reshape(-1)
    finite = hours[np.isfinite(hours)] if hours.size else hours
    observed: List[str]
    if finite.size < 3:
        observed = ["last observed: not stratified"] * len(frame)
    else:
        low, high = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
        if not (low < high):
            observed = ["last observed: not stratified"] * len(frame)
        else:
            observed = [
                "last observed: unknown" if not np.isfinite(value)
                else "last observed: nearest delivery" if value <= low
                else "last observed: middle" if value <= high
                else "last observed: earliest"
                for value in hours.tolist()
            ]
    return {"ascertainment": strata, "coverage": observed}


def figure_coverage_space(
    frame: pd.DataFrame,
    values: Any,
    projection: Any,
    directory: Any,
    *,
    groupings: Optional[Mapping[str, Sequence[str]]] = None,
) -> Path:
    """Draw the same map, coloured by ascertainment and by coverage instead of by outcome.

    One model version, because the question is not before-and-after: it is whether the separation a
    reader sees in Figure 1 tracks how these recordings were selected and how much of them survived.
    It is drawn on the **same** saved projection, so the two figures are the same coordinates.

    Args:
        frame: The recording table the values belong to.
        values: Standardized recording vectors, ``(N, d_z)``.
        projection: The shared projection.
        directory: The run directory.
        groupings: Grouping name -> one label per row. ``None`` uses
            :func:`coverage_groupings`.

    Returns:
        The written path.
    """
    figures = _figures()
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame):
        raise PilotConfigError(
            f"the coverage view was handed {matrix.shape} values for {len(frame)} row(s)."
        )
    grouped = dict(groupings if groupings is not None else coverage_groupings(frame))
    points = projection.transform(matrix)
    palette = _palette([label for values_ in grouped.values() for label in values_])

    fig, axes = figures.new_figure(1, max(len(grouped), 1), height_per_row=4.2, width=9.0)
    for column, (name, labels) in enumerate(grouped.items()):
        _scatter_panel(
            axes[0, column], points, labels,
            palette=palette, projection=projection,
            title=f"coloured by {name}",
        )
    fig.suptitle(f"The same map, by ascertainment and coverage -- {variance_caption(projection)}")
    fig.text(0.01, 0.005, CAPTIONS[FIGURE_COVERAGE_SPACE], fontsize=6.0, wrap=True)
    return figures.render_figure(fig, _figure_dir(directory) / FIGURE_COVERAGE_SPACE)


# =============================================================================
# Figure 2 -- the supervised axis
# =============================================================================
def figure_supervised_axis(
    frame: pd.DataFrame,
    columns: Mapping[str, Any],
    directory: Any,
    *,
    metrics: Optional[Mapping[str, Mapping[str, Any]]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> Path:
    """Draw each model's held-out score distribution, split by clinical class.

    One panel per model, and each panel's y-axis is that model's **own** logit scale -- which is why
    the panels are not given shared limits and why every one of them carries
    :func:`score_scale_note`. The threshold drawn on each is the one chosen on validation.

    Args:
        frame: One row per held-out recording, carrying the GUID, the outcome and the class name.
        columns: ``{model name: logits}``, each aligned with ``frame``.
        directory: The run directory.
        metrics: ``{model name: recording_metrics record}``, printed in the panel titles. Absent
            metrics leave the title without them rather than with invented ones.
        thresholds: ``{model name: threshold}``, drawn as a reference line where supplied.

    Returns:
        The written path.

    Raises:
        PilotConfigError: If a model's scores do not align with the frame.
    """
    figures = _figures()
    classes = _class_of(frame)
    palette = _palette(classes)
    order = [label for label in PLOT_CLASSES if label in set(classes)]
    order += sorted(set(classes) - set(order))

    names = list(columns)
    fig, axes = figures.new_figure(max(len(names), 1), 1, height_per_row=3.2, width=9.0)
    for row, name in enumerate(names):
        values = np.asarray(columns[name], dtype=np.float64).reshape(-1)
        if values.size != len(frame):
            raise PilotConfigError(
                f"model {name!r} supplies {values.size} score(s) for {len(frame)} recording(s); a "
                f"panel drawn from that would colour one recording with another's class."
            )
        samples = {
            f"{label or 'unlabelled'} (n={sum(1 for value in classes if value == label)})": values[
                [index for index, value in enumerate(classes) if value == label]
            ]
            for label in order
        }
        measured = dict((metrics or {}).get(name) or {})
        headline = " ".join(
            part for part in (
                f"AUROC {_num(measured.get('auroc'), 3)}",
                f"AP {_num(measured.get('average_precision'), 3)}",
                f"(prevalence {_num(measured.get('prevalence'), 3)})",
                f"n={_int(measured.get('n_recordings'))}",
            ) if part
        ) if measured else "metrics not supplied"
        figures.violin_panel(
            axes[row, 0],
            samples,
            title=f"{name}: supervised axis -- {headline}",
            ylabel="w'S(v) + b",
            colors={
                key: palette.get(key.split(" (")[0], figures.COLOR_GRAY) for key in samples
            },
            reference=None if thresholds is None else thresholds.get(name),
            reference_label="validation threshold",
        )
        # On the axis label rather than floating below the axes: an annotation outside the axes
        # is invisible to tight_layout and lands on the next panel's title.
        axes[row, 0].set_xlabel(score_scale_note(name, short=True))

    fig.suptitle("Separation along each model's own supervised direction")
    fig.text(0.01, 0.005, CAPTIONS[FIGURE_SUPERVISED_AXIS], fontsize=6.0, wrap=True)
    return figures.render_figure(fig, _figure_dir(directory) / FIGURE_SUPERVISED_AXIS)


# =============================================================================
# Figure 3 -- the last three hours
# =============================================================================
def signed_bin_hours(*, bin_hours: float, window_hours: float) -> np.ndarray:
    """The x coordinate of each trajectory bin, signed so delivery sits at zero on the right.

    Named for the **analysis** window rather than the preservation one: a run may analyse further
    back than its objective preserves, and an argument named after the narrower setting would
    describe an axis it does not span.

    Args:
        bin_hours: Bin width.
        window_hours: The analysis window's upper edge, from
            :func:`~latent_pilot.config.analysis_hours`.

    Returns:
        One negative midpoint per bin, indexed the way
        :func:`~latent_pilot.data.bin_edges` indexes them -- entry 0 is the bin nearest delivery.
    """
    edges = data.bin_edges(bin_hours=bin_hours, preservation_hours=window_hours)
    return np.asarray([-0.5 * (low + high) for low, high in edges], dtype=np.float64)


def _band_curves(
    bands: pd.DataFrame, x: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """Reshape a band table into per-group arrays over the full bin axis, gaps included.

    A bin a group never occupied stays ``nan`` here, which matplotlib draws as a break in the line.
    Filling it would draw a measurement that was never made.

    Args:
        bands: The table from :func:`~latent_pilot.analyze.group_bands`.
        x: The bin axis, whose length fixes the array length.

    Returns:
        ``{group: {'mean': ..., 'lo': ..., 'hi': ..., 'n': ...}}``.
    """
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for _index, row in bands.iterrows():
        group = str(row["group"])
        entry = curves.setdefault(group, {
            "mean": np.full(x.size, np.nan),
            "lo": np.full(x.size, np.nan),
            "hi": np.full(x.size, np.nan),
            "n": np.zeros(x.size, dtype=np.int64),
        })
        position = int(row[data.BIN_COLUMN])
        if not 0 <= position < x.size:
            continue
        entry["mean"][position] = float(row["mean"])
        entry["lo"][position] = float(row["lo"])
        entry["hi"][position] = float(row["hi"])
        entry["n"][position] = int(row["n_recordings"])
    return curves


def figure_trajectories(
    versions: Mapping[str, Tuple[pd.DataFrame, pd.DataFrame]],
    directory: Any,
    *,
    bin_hours: float,
    window_hours: float,
    supervised_hours: float,
    n_traces: int = DEFAULT_N_TRACES,
    seed: int = 0,
    excerpts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Path:
    """Draw group means, bands, counts and a seeded sample of individual traces per model.

    Args:
        versions: ``{name: (bands, scored)}`` -- the band table from
            :func:`~latent_pilot.analyze.group_bands` and the scored bin table from
            :func:`~latent_pilot.analyze.score_frame`, per model.
        directory: The run directory.
        bin_hours: Bin width, for the x axis.
        window_hours: The analysis window's upper edge; the left end of the axis.
        supervised_hours: The supervised window, shaded and labelled -- every bin outside it is an
            application of the head outside the window it was fitted on.
        n_traces: How many individual recordings to overlay. The **same** recordings in every
            panel, drawn once from the first model's table, so a trace can be followed across
            models.
        seed: Seeds the trace selection.
        excerpts: Optional ``{guid: {'hours': ..., 'fhr': ..., 'up': ...}}`` raw signal excerpts,
            each on the same signed-hour axis. Recordings without an excerpt are simply not drawn
            one; nothing is resampled or extended to fill the panel.

    Returns:
        The written path.
    """
    figures = _figures()
    x = signed_bin_hours(bin_hours=bin_hours, window_hours=window_hours)
    names = list(versions)
    palette = _palette(
        [group for name in names for group in versions[name][0].get("group", [])]
        + [label for name in names for label in _class_of(versions[name][1])]
    )

    traces: List[str] = []
    if names and int(n_traces) > 0:
        _bands, first_scored = versions[names[0]]
        unique = first_scored.drop_duplicates(subset=[data.GUID_COLUMN])
        traces = select_traces(unique, n=int(n_traces), seed=int(seed))

    excerpt_guids = [guid for guid in traces if guid in dict(excerpts or {})]
    fig, axes = figures.new_figure(
        max(len(names), 1) + len(excerpt_guids), 1, height_per_row=3.0, width=9.0
    )

    for row, name in enumerate(names):
        ax = axes[row, 0]
        bands, scored = versions[name]
        curves = _band_curves(bands, x)

        ax.axvspan(
            -float(supervised_hours), 0.0,
            color=figures.COLOR_LIGHT_GRAY, alpha=0.5, linewidth=0.0,
            label=f"supervised window (final {supervised_hours:g} h)",
        )
        for guid in traces:
            block = scored[scored[data.GUID_COLUMN].astype(str) == str(guid)]
            if block.empty:
                continue
            trace = np.full(x.size, np.nan)
            for position, score in zip(
                block[data.BIN_COLUMN].tolist(),
                np.asarray(block[analyze.SCORE_COLUMN], dtype=np.float64).tolist(),
            ):
                if 0 <= int(position) < x.size:
                    trace[int(position)] = float(score)
            if not np.isfinite(trace).any():
                continue
            colour = palette.get(_class_of(block)[0], figures.COLOR_GRAY)
            ax.plot(
                x, trace, color=colour, alpha=0.35, linewidth=0.6,
                marker="o", markersize=1.8,
            )

        for offset, group in enumerate(sorted(curves)):
            entry = curves[group]
            colour = palette.get(str(group), figures.COLOR_GRAY)
            ax.fill_between(
                x, entry["lo"], entry["hi"], color=colour, alpha=0.2, linewidth=0.0,
            )
            ax.plot(x, entry["mean"], color=colour, marker="o", markersize=3.0, label=str(group))
            # Counts inside the axes, one row per group: below the frame they would collide with
            # the tick labels, and tight_layout does not know they are there.
            for position, count in enumerate(entry["n"].tolist()):
                ax.annotate(
                    str(int(count)),
                    xy=(x[position], 0.03 + 0.06 * offset),
                    xycoords=("data", "axes fraction"),
                    ha="center", fontsize=5.5, color=colour,
                )

        ax.set_xlim(-float(window_hours), 0.0)
        ax.set_title(f"{name}: frozen final-hour head applied to every occupied bin")
        ax.set_xlabel(
            f"hours before delivery (delivery at 0) -- {score_scale_note(name, short=True)}"
        )
        ax.set_ylabel("w'S(v) + b")
        ax.legend(loc="best", fontsize=6.0)
        figures.style_axes(ax)

    for offset, guid in enumerate(excerpt_guids):
        ax = axes[len(names) + offset, 0]
        excerpt = dict(dict(excerpts or {})[guid])
        hours = np.asarray(excerpt.get("hours", []), dtype=np.float64).reshape(-1)
        fhr = np.asarray(excerpt.get("fhr", []), dtype=np.float64).reshape(-1)
        up = np.asarray(excerpt.get("up", []), dtype=np.float64).reshape(-1)
        if hours.size and fhr.size == hours.size:
            ax.plot(hours, fhr, color=figures.COLOR_BLUE, linewidth=0.6, label="FHR")
        if hours.size and up.size == hours.size:
            twin = ax.twinx()
            twin.plot(hours, up, color=figures.COLOR_ORANGE, linewidth=0.6, label="UP")
            twin.set_ylabel("UP")
        ax.set_xlim(-float(window_hours), 0.0)
        ax.set_title(f"raw excerpt -- {guid}")
        ax.set_xlabel("hours before delivery (delivery at 0)")
        ax.set_ylabel("FHR")
        figures.style_axes(ax)

    fig.suptitle(
        f"Score over the last {float(window_hours):g} hours, "
        f"with per-bin recording counts"
    )
    fig.text(
        0.01, 0.005,
        CAPTIONS[FIGURE_TRAJECTORIES].format(
            supervised_hours=float(supervised_hours), bin_hours=float(bin_hours)
        ),
        fontsize=6.0, wrap=True,
    )
    logger.info(
        f"figure 3: {len(names)} model panel(s), {len(traces)} seeded individual trace(s), "
        f"{len(excerpt_guids)} raw excerpt(s)"
    )
    return figures.render_figure(fig, _figure_dir(directory) / FIGURE_TRAJECTORIES)


# =============================================================================
# Figures 4-8 -- the classification result, overall and against time
# =============================================================================
#: How many small-multiple ROC panels are placed on one row.
ROC_GRID_COLUMNS = 4

#: The metrics Figure 6 draws against time, one panel each, in this order.
TIME_METRICS: Tuple[str, ...] = ("auroc", "average_precision", "f1", "balanced_accuracy")

#: How each of those panels marks the level a useless ranker would reach. ``prevalence`` names the
#: per-bin column to read, because average precision has no fixed chance level; ``None`` means the
#: metric has no meaningful chance line and drawing one would invent a reference.
TIME_METRIC_CHANCE: Dict[str, Any] = {
    "auroc": 0.5,
    "average_precision": "prevalence",
    "f1": None,
    "balanced_accuracy": 0.5,
}

#: The confusion cells Figure 8 stacks, outermost first, with the label each carries.
COUNT_STACK: Tuple[Tuple[str, str], ...] = (
    ("tp", "true positive"),
    ("fn", "false negative"),
    ("fp", "false positive"),
    ("tn", "true negative"),
)


#: Share of the figure height reserved below the axes for the caption. Figures 1-3 are single
#: columns of tall panels, where a caption at the very bottom clears the last x-label on its own.
#: These are grids, whose bottom row of x-labels sits far lower, so the space is reserved
#: explicitly rather than left to collide.
CAPTION_BAND = 0.05


def _render_with_caption(fig: Any, directory: Any, stem: str) -> Path:
    """Lay the figure out above its caption, write the caption, and save.

    ``tight_layout`` knows nothing about a ``fig.text`` placed in figure coordinates, so it is
    given an explicit rectangle to lay the axes into and :func:`render_figure` is asked not to run
    it a second time -- which would undo the reservation.

    Args:
        fig: The figure.
        directory: The run directory.
        stem: The figure-name constant, which is also its :data:`CAPTIONS` key.

    Returns:
        The path written, extension included.
    """
    figures = _figures()
    try:
        fig.tight_layout(rect=(0.0, CAPTION_BAND, 1.0, 1.0))
    except Exception:  # noqa: BLE001 - a layout warning must not lose a completed figure
        pass
    fig.text(0.01, 0.005, CAPTIONS[stem], fontsize=6.0, wrap=True)
    return figures.render_figure(fig, _figure_dir(directory) / stem, tight=False)


def _empty_panel(ax: Any, message: str) -> None:
    """Say why a panel is blank, rather than leaving a reader to guess.

    Args:
        ax: Target axes.
        message: The reason, short enough to sit inside the frame.
    """
    figures = _figures()
    ax.text(
        0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
        fontsize=6.0, color=figures.COLOR_GRAY, wrap=True,
    )
    figures.style_axes(ax)


def _interval_text(
    intervals: Optional[Mapping[str, Any]], model: str, metric: str, digits: int = 3
) -> str:
    """The bracketed interval for one model and metric, or an empty string when there is none.

    Args:
        intervals: ``paired_bootstrap``'s ``models`` block, or ``None``.
        model: The model name.
        metric: The metric name.
        digits: Decimals.

    Returns:
        ``' [lo, hi]'`` or ``''``. Never a bracket around ``nan``: an interval the estimator
        declined to form is absent from the legend rather than printed as an empty range.
    """
    record = dict((intervals or {}).get(model) or {}).get(metric)
    if not isinstance(record, Mapping):
        return ""
    low, high = record.get("lo"), record.get("hi")
    if low is None or high is None or not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return ""
    return f" [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def figure_roc_pr(
    curves: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    directory: Any,
    *,
    intervals: Optional[Mapping[str, Any]] = None,
) -> Path:
    """ROC and precision-recall for the held-out final-hour bags, both models on one pair of axes.

    Two panels rather than two figures: the models are compared, and a comparison split across
    files is one a reader has to assemble. Both panels carry the level a useless ranker reaches --
    the diagonal for the ROC, the adverse-outcome prevalence for the PR curve, which unlike the
    diagonal moves with the cohort and is therefore drawn from the data rather than assumed.

    Each model's **validation-selected operating point** is marked on both curves, at
    ``(1 - specificity, sensitivity)`` and ``(sensitivity, precision)``. That point is the only
    place a threshold enters this figure; the curves themselves are threshold-free, which is why
    they can be compared between two models whose logit scales are not comparable.

    Args:
        curves: ``{model: {'roc': ..., 'pr': ...}}`` as
            :func:`~latent_pilot.evaluate.roc_points` and
            :func:`~latent_pilot.evaluate.pr_points` return them.
        metrics: ``{model: recording_metrics}``, for the operating point and the counts.
        directory: The run directory; the figure lands in its ``figures`` subdirectory.
        intervals: ``paired_bootstrap``'s ``models`` block, for the legend intervals. Optional --
            a run whose bootstrap could not be estimated still gets the curves.

    Returns:
        The path written.
    """
    figures = _figures()
    names = [str(name) for name in curves]
    palette = _palette(names)
    fig, axes = figures.new_figure(1, 2, height_per_row=3.4, width=9.0)

    roc_ax, pr_ax = axes[0, 0], axes[0, 1]
    roc_ax.plot([0.0, 1.0], [0.0, 1.0], color=figures.COLOR_GRAY, linewidth=0.8, linestyle="--")
    drawn_roc = 0
    for name in names:
        record = dict(dict(curves[name]).get("roc") or {})
        false_positive = np.asarray(record.get("fpr", []), dtype=np.float64)
        true_positive = np.asarray(record.get("tpr", []), dtype=np.float64)
        if false_positive.size == 0:
            continue
        colour = palette.get(name, figures.COLOR_GRAY)
        label = (
            f"{name}: AUROC {float(record.get('auroc', float('nan'))):.3f}"
            f"{_interval_text(intervals, name, 'auroc')}"
        )
        roc_ax.plot(false_positive, true_positive, color=colour, linewidth=1.2, label=label)
        measured = dict(metrics.get(name) or {})
        specificity, sensitivity = measured.get("specificity"), measured.get("sensitivity")
        if specificity is not None and sensitivity is not None and np.isfinite(
            float(specificity)
        ) and np.isfinite(float(sensitivity)):
            roc_ax.plot(
                [1.0 - float(specificity)], [float(sensitivity)],
                marker="o", markersize=4.0, color=colour, linestyle="none",
            )
        drawn_roc += 1
    if not drawn_roc:
        _empty_panel(roc_ax, "no held-out population carried both classes")
    else:
        roc_ax.set_xlim(0.0, 1.0)
        roc_ax.set_ylim(0.0, 1.0)
        roc_ax.set_title("ROC, held-out final-hour bags")
        roc_ax.set_xlabel("false positive rate (1 - specificity)")
        roc_ax.set_ylabel("true positive rate (sensitivity)")
        roc_ax.legend(loc="lower right", fontsize=6.0)
        figures.style_axes(roc_ax)

    drawn_pr = 0
    prevalence = float("nan")
    for name in names:
        record = dict(dict(curves[name]).get("pr") or {})
        recall = np.asarray(record.get("recall", []), dtype=np.float64)
        precision = np.asarray(record.get("precision", []), dtype=np.float64)
        if recall.size == 0:
            continue
        prevalence = float(record.get("prevalence", float("nan")))
        colour = palette.get(name, figures.COLOR_GRAY)
        label = (
            f"{name}: AP {float(record.get('average_precision', float('nan'))):.3f}"
            f"{_interval_text(intervals, name, 'average_precision')}"
        )
        # ``post`` because precision-recall is a step function of the threshold: joining the
        # points with straight lines draws interpolated operating rules that do not exist.
        pr_ax.step(recall, precision, where="post", color=colour, linewidth=1.2, label=label)
        measured = dict(metrics.get(name) or {})
        sensitivity, positive_predictive = measured.get("sensitivity"), measured.get("precision")
        if sensitivity is not None and positive_predictive is not None and np.isfinite(
            float(sensitivity)
        ) and np.isfinite(float(positive_predictive)):
            pr_ax.plot(
                [float(sensitivity)], [float(positive_predictive)],
                marker="o", markersize=4.0, color=colour, linestyle="none",
            )
        drawn_pr += 1
    if not drawn_pr:
        _empty_panel(pr_ax, "no held-out population carried both classes")
    else:
        if np.isfinite(prevalence):
            pr_ax.axhline(
                prevalence, color=figures.COLOR_GRAY, linewidth=0.8, linestyle="--",
                label=f"chance = prevalence {prevalence:.3f}",
            )
        pr_ax.set_xlim(0.0, 1.0)
        pr_ax.set_ylim(0.0, 1.0)
        pr_ax.set_title("Precision-recall, held-out final-hour bags")
        pr_ax.set_xlabel("recall (sensitivity)")
        pr_ax.set_ylabel("precision")
        pr_ax.legend(loc="best", fontsize=6.0)
        figures.style_axes(pr_ax)

    counts = dict(metrics.get(names[0]) or {}) if names else {}
    fig.suptitle(
        f"Held-out discrimination: {int(counts.get('n_recordings', 0))} recording(s), "
        f"{int(counts.get('n_adverse', 0))} adverse; dots mark the validation-selected threshold"
    )
    logger.info(f"figure 4: {drawn_roc} ROC curve(s), {drawn_pr} PR curve(s)")
    return _render_with_caption(fig, directory, FIGURE_ROC_PR)


def figure_confusion(
    metrics: Mapping[str, Mapping[str, Any]],
    directory: Any,
    *,
    intervals: Optional[Mapping[str, Any]] = None,
) -> Path:
    """The confusion matrix of every model at its validation threshold, and the rates beside it.

    The counts and the rates share a figure because neither is readable alone: four cells without
    the rates hide how the two classes were traded off, and rates without the cells hide that a
    sensitivity of one may rest on two recordings.

    All panels share one colour scale, so a cell of the same colour is the same count in every
    matrix -- two matrices each scaled to their own maximum would paint two different counts
    identically and make the comparison the figure exists for the one thing it cannot support.

    Args:
        metrics: ``{model: recording_metrics}``.
        directory: The run directory.
        intervals: ``paired_bootstrap``'s ``models`` block, for the rate intervals. Optional.

    Returns:
        The path written.
    """
    figures = _figures()
    names = [str(name) for name in metrics]
    fig, axes = figures.new_figure(1, max(len(names) + 1, 2), height_per_row=3.2, width=9.0)

    fields = {
        name: np.asarray(
            [
                [float(dict(metrics[name]).get("tn", np.nan)),
                 float(dict(metrics[name]).get("fp", np.nan))],
                [float(dict(metrics[name]).get("fn", np.nan)),
                 float(dict(metrics[name]).get("tp", np.nan))],
            ],
            dtype=np.float64,
        )
        for name in names
    }
    finite = np.concatenate([field[np.isfinite(field)].ravel() for field in fields.values()]) \
        if fields else np.asarray([], dtype=np.float64)
    limits = (0.0, float(finite.max())) if finite.size and finite.max() > 0 else None

    for column, name in enumerate(names):
        ax = axes[0, column]
        field = fields[name]
        figures.heatmap_with_colorbar(
            fig, ax, field,
            title=f"{name} at threshold {float(dict(metrics[name]).get('threshold', np.nan)):.3f}",
            xlabel="predicted", ylabel="actual",
            symmetric=False, vlimits=limits, colorbar_label="recordings",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["healthy", "adverse"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["healthy", "adverse"])
        for row in range(2):
            for cell in range(2):
                value = field[row, cell]
                ax.text(
                    cell, row, "-" if not np.isfinite(value) else str(int(value)),
                    ha="center", va="center", fontsize=8.0, color=figures.COLOR_BLACK,
                )

    ax = axes[0, len(names)]
    lines: List[str] = []
    for name in names:
        measured = dict(metrics[name])
        lines.append(name)
        for metric in (
            "sensitivity", "specificity", "precision", "npv", "f1", "accuracy",
            "balanced_accuracy",
        ):
            value = measured.get(metric)
            text = (
                MISSING if value is None or not np.isfinite(float(value))
                else f"{float(value):.3f}"
            )
            lines.append(f"  {metric}: {text}{_interval_text(intervals, name, metric)}")
        lines.append("")
    ax.axis("off")
    ax.text(
        0.0, 1.0, "\n".join(lines) or MISSING, transform=ax.transAxes,
        ha="left", va="top", fontsize=6.0, family="monospace",
    )

    for column in range(len(names) + 1, axes.shape[1]):
        axes[0, column].axis("off")

    fig.suptitle("Confusion counts and derived rates at the validation-selected threshold")
    logger.info(f"figure 5: {len(names)} confusion matrix panel(s)")
    return _render_with_caption(fig, directory, FIGURE_CONFUSION)


def _metric_curve(
    block: pd.DataFrame, metric: str, x: np.ndarray
) -> Dict[str, np.ndarray]:
    """One model's curve for one metric over the full bin axis, absences left absent.

    A bin the model has no estimable cell in stays ``nan``, which matplotlib draws as a break.
    Filling it would draw a discrimination that was never measured -- and the bins most likely to
    be unestimable are the early ones, which is exactly where a filled line would invent a trend.

    Args:
        block: One model's rows of the per-bin table.
        metric: The metric column.
        x: The bin axis, whose length fixes the array length.

    Returns:
        ``{'mean', 'lo', 'hi', 'n', 'n_adverse', 'chance'}``, all of length ``x.size``.
    """
    curve = {
        "mean": np.full(x.size, np.nan),
        "lo": np.full(x.size, np.nan),
        "hi": np.full(x.size, np.nan),
        "chance": np.full(x.size, np.nan),
        "n": np.zeros(x.size, dtype=np.int64),
        "n_adverse": np.zeros(x.size, dtype=np.int64),
    }
    for _index, row in block.iterrows():
        position = int(row[data.BIN_COLUMN])
        if not 0 <= position < x.size:
            continue
        curve["n"][position] = int(row.get("n_recordings", 0))
        curve["n_adverse"][position] = int(row.get("n_adverse", 0))
        if not bool(row.get("estimable", False)):
            # The chance level stays absent here too. A bin with no adverse recording has a
            # prevalence of exactly zero, and a reference line dropping to zero beside a curve
            # that was not drawn reads as a measurement rather than as a missing one.
            continue
        curve["chance"][position] = float(row.get("prevalence", np.nan))
        curve["mean"][position] = float(row.get(metric, np.nan))
        curve["lo"][position] = float(row.get(f"{metric}_lo", np.nan))
        curve["hi"][position] = float(row.get(f"{metric}_hi", np.nan))
    return curve


def figure_metrics_vs_time(
    bin_metrics: pd.DataFrame,
    directory: Any,
    *,
    bin_hours: float,
    window_hours: float,
    supervised_hours: float,
    metrics: Sequence[str] = TIME_METRICS,
) -> Path:
    """Discrimination as a function of time before delivery, one panel per metric.

    The frozen final-hour head is applied to every bin, so only the bins inside the shaded window
    are the rule evaluated where its loss was defined; everything left of it is an application
    outside that window, which is the interesting part and also the part a reader must not mistake
    for a fitted result.

    Counts sit under every bin because the cohort is **whoever was observed there**: a metric that
    rises towards delivery may be a real trend or a change in who was still being recorded, and
    only the counts let a reader tell the two apart.

    Args:
        bin_metrics: The table from :func:`~latent_pilot.analyze.bin_classification`.
        directory: The run directory.
        bin_hours: Bin width.
        window_hours: The analysis window's upper edge, which fixes the axis.
        supervised_hours: The supervised window, shaded.
        metrics: Which metrics get a panel.

    Returns:
        The path written.
    """
    figures = _figures()
    wanted = [str(metric) for metric in metrics]
    x = signed_bin_hours(bin_hours=bin_hours, window_hours=window_hours)
    names = (
        sorted({str(value) for value in bin_metrics["model"]})
        if not bin_metrics.empty else []
    )
    palette = _palette(names)
    columns = 2 if len(wanted) > 1 else 1
    rows = int(np.ceil(len(wanted) / columns))
    fig, axes = figures.new_figure(rows, columns, height_per_row=2.8, width=9.0)

    for position, metric in enumerate(wanted):
        ax = axes[position // columns, position % columns]
        ax.axvspan(
            -float(supervised_hours), 0.0, color=figures.COLOR_LIGHT_GRAY, alpha=0.5,
            linewidth=0.0,
        )
        drawn = 0
        for offset, name in enumerate(names):
            block = bin_metrics[bin_metrics["model"] == name]
            curve = _metric_curve(block, metric, x)
            if not np.isfinite(curve["mean"]).any():
                continue
            colour = palette.get(name, figures.COLOR_GRAY)
            ax.fill_between(
                x, curve["lo"], curve["hi"], color=colour, alpha=0.2, linewidth=0.0,
            )
            ax.plot(x, curve["mean"], color=colour, marker="o", markersize=3.0, label=name)
            for index, count in enumerate(curve["n"].tolist()):
                if not count:
                    continue
                ax.annotate(
                    f"{int(curve['n_adverse'][index])}/{int(count)}",
                    xy=(x[index], 0.03 + 0.06 * offset),
                    xycoords=("data", "axes fraction"),
                    ha="center", fontsize=5.0, color=colour,
                )
            drawn += 1
        chance = TIME_METRIC_CHANCE.get(metric)
        if isinstance(chance, (int, float)):
            ax.axhline(
                float(chance), color=figures.COLOR_GRAY, linewidth=0.8, linestyle="--",
            )
        elif chance == "prevalence" and names:
            reference = _metric_curve(
                bin_metrics[bin_metrics["model"] == names[0]], metric, x
            )["chance"]
            ax.plot(
                x, reference, color=figures.COLOR_GRAY, linewidth=0.8, linestyle="--",
                label="chance = prevalence",
            )
        if not drawn:
            _empty_panel(ax, f"no bin carried both classes for {metric}")
            continue
        ax.set_xlim(-float(window_hours), 0.0)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("hours before delivery (delivery at 0)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(loc="best", fontsize=6.0)
        figures.style_axes(ax)

    for position in range(len(wanted), rows * columns):
        axes[position // columns, position % columns].axis("off")

    fig.suptitle(
        f"Discrimination over the last {float(window_hours):g} hours; "
        f"counts are adverse/total recordings per bin"
    )
    logger.info(f"figure 6: {len(wanted)} metric panel(s) over {len(names)} model(s)")
    return _render_with_caption(fig, directory, FIGURE_METRICS_TIME)


def figure_roc_by_bin(
    bin_curves: pd.DataFrame,
    bin_metrics: pd.DataFrame,
    directory: Any,
    *,
    n_cols: int = ROC_GRID_COLUMNS,
) -> Path:
    """One ROC panel per time bin, both models overlaid, nearest delivery first.

    Only bins that carried both classes appear: a bin with no adverse recording has no ROC, and a
    panel drawn for it would show a diagonal that was never measured. The panel count is therefore
    itself a statement about coverage, and the title of each panel carries the counts it rests on.

    Args:
        bin_curves: The long curve table from :func:`~latent_pilot.analyze.bin_roc_points`.
        bin_metrics: The per-bin table, read for each panel's counts and label.
        directory: The run directory.
        n_cols: Panels per row.

    Returns:
        The path written.
    """
    figures = _figures()
    if bin_curves.empty:
        fig, axes = figures.new_figure(1, 1, height_per_row=3.0, width=9.0)
        _empty_panel(axes[0, 0], "no time bin carried both binary classes")
        fig.suptitle("ROC by time bin")
        return _render_with_caption(fig, directory, FIGURE_ROC_BINS)

    bins = sorted({int(value) for value in bin_curves[data.BIN_COLUMN]})
    names = sorted({str(value) for value in bin_curves["model"]})
    palette = _palette(names)
    columns = int(min(max(n_cols, 1), len(bins)))
    rows = int(np.ceil(len(bins) / columns))
    fig, axes = figures.new_figure(rows, columns, height_per_row=2.4, width=9.0)

    for position, index in enumerate(bins):
        ax = axes[position // columns, position % columns]
        ax.plot([0.0, 1.0], [0.0, 1.0], color=figures.COLOR_GRAY, linewidth=0.7, linestyle="--")
        label = ""
        counts = ""
        for name in names:
            block = bin_curves[
                (bin_curves["model"] == name) & (bin_curves[data.BIN_COLUMN] == index)
            ].sort_values("point")
            if block.empty:
                continue
            label = str(block[data.BIN_LABEL_COLUMN].iloc[0])
            counts = (
                f"{int(block['n_adverse'].iloc[0])}/{int(block['n_recordings'].iloc[0])}"
            )
            ax.plot(
                block["fpr"].to_numpy(), block["tpr"].to_numpy(),
                color=palette.get(name, figures.COLOR_GRAY), linewidth=1.0,
                label=f"{name} {float(block['auroc'].iloc[0]):.2f}",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{label} h before delivery, {counts}", fontsize=7.0)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right", fontsize=5.5)
        figures.style_axes(ax)

    for position in range(len(bins), rows * columns):
        axes[position // columns, position % columns].axis("off")

    supervised = ""
    if not bin_metrics.empty and "supervised_window" in bin_metrics.columns:
        inside = sorted({
            int(row[data.BIN_COLUMN]) for _index, row in bin_metrics.iterrows()
            if bool(row["supervised_window"])
        })
        supervised = f"; bins {inside} lie inside the supervised window" if inside else ""
    fig.suptitle(
        f"ROC in each time bin, nearest delivery first; titles carry adverse/total{supervised}"
    )
    logger.info(f"figure 7: {len(bins)} estimable bin panel(s)")
    return _render_with_caption(fig, directory, FIGURE_ROC_BINS)


def figure_counts_vs_time(
    bin_metrics: pd.DataFrame,
    directory: Any,
    *,
    bin_hours: float,
    window_hours: float,
    supervised_hours: float,
) -> Path:
    """The four confusion cells per time bin, stacked, one row per model.

    Where Figure 6 shows rates, this shows what they are rates **of**. A sensitivity that improves
    towards delivery while the adverse count falls to two is a different finding from one measured
    on thirty, and a stacked count is the least interpretable-away way to show it.

    Args:
        bin_metrics: The table from :func:`~latent_pilot.analyze.bin_classification`.
        directory: The run directory.
        bin_hours: Bin width, which also fixes the bar width.
        window_hours: The analysis window's upper edge.
        supervised_hours: The supervised window, shaded.

    Returns:
        The path written.
    """
    figures = _figures()
    names = (
        sorted({str(value) for value in bin_metrics["model"]})
        if not bin_metrics.empty else []
    )
    x = signed_bin_hours(bin_hours=bin_hours, window_hours=window_hours)
    palette = _palette([label for _cell, label in COUNT_STACK])
    fig, axes = figures.new_figure(max(len(names), 1), 1, height_per_row=2.6, width=9.0)

    if not names:
        _empty_panel(axes[0, 0], "no model produced a per-bin cell")
    for row, name in enumerate(names):
        ax = axes[row, 0]
        ax.axvspan(
            -float(supervised_hours), 0.0, color=figures.COLOR_LIGHT_GRAY, alpha=0.5,
            linewidth=0.0,
        )
        block = bin_metrics[bin_metrics["model"] == name]
        heights = {cell: np.zeros(x.size, dtype=np.float64) for cell, _label in COUNT_STACK}
        for _index, entry in block.iterrows():
            position = int(entry[data.BIN_COLUMN])
            if not 0 <= position < x.size:
                continue
            for cell, _label in COUNT_STACK:
                heights[cell][position] = float(entry.get(cell, 0.0))
        bottom = np.zeros(x.size, dtype=np.float64)
        for cell, label in COUNT_STACK:
            ax.bar(
                x, heights[cell], bottom=bottom, width=0.8 * float(bin_hours),
                color=palette.get(label, figures.COLOR_GRAY), label=label, linewidth=0.0,
            )
            bottom = bottom + heights[cell]
        ax.set_xlim(-float(window_hours), 0.0)
        ax.set_title(f"{name}: confusion cells per bin at its validation threshold")
        ax.set_xlabel("hours before delivery (delivery at 0)")
        ax.set_ylabel("recordings")
        ax.legend(loc="upper left", fontsize=6.0, ncol=4)
        figures.style_axes(ax)

    fig.suptitle(
        f"What the rates are rates of, over the last {float(window_hours):g} hours"
    )
    logger.info(f"figure 8: {len(names)} model panel(s)")
    return _render_with_caption(fig, directory, FIGURE_COUNTS_TIME)


# =============================================================================
# Rendering measured values, and refusing to render unmeasured ones
# =============================================================================
def _num(value: Any, digits: int = 4) -> str:
    """One number, or :data:`MISSING` when there is not one.

    Args:
        value: The measured value, possibly ``None`` or ``nan``.
        digits: Significant decimals.

    Returns:
        The formatted number, or :data:`MISSING`. ``nan`` is never printed as ``'nan'``: a reader
        skimming a table reads that as a number that came out strange rather than as one that was
        never established.
    """
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{int(digits)}f}" if np.isfinite(number) else MISSING


def _int(value: Any) -> str:
    """One count, or :data:`MISSING`."""
    if value is None:
        return MISSING
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return MISSING


def _text(value: Any) -> str:
    """One free-text field, or :data:`MISSING` when it is absent or empty.

    A mapping -- the software record is one -- is flattened to ``key=value`` pairs rather than
    printed as a Python literal, so a reader can read the revision out of the table.
    """
    if value is None:
        return MISSING
    if isinstance(value, Mapping):
        rendered = ", ".join(
            f"{key}={MISSING if item is None else item}" for key, item in sorted(value.items())
        )
    else:
        rendered = str(value).strip()
    return rendered if rendered else MISSING


def _ci(record: Optional[Mapping[str, Any]], digits: int = 4) -> str:
    """A point estimate with its interval, or the estimator's own reason for having none.

    Args:
        record: An interval record carrying ``point``, ``lo``, ``hi`` and optionally ``note`` --
            the shape :func:`~teb_vae.lag_attn.eval.stats.bootstrap_ci` and the paired bootstrap
            both return.
        digits: Significant decimals.

    Returns:
        ``'0.7123 [0.6011, 0.8240]'``, or the point with the note that explains the absent bounds,
        or :data:`MISSING`.
    """
    if not record:
        return MISSING
    point = _num(record.get("point"), digits)
    bounds = [_num(record.get(key), digits) for key in ("lo", "hi")]
    if MISSING in bounds:
        note = _text(record.get("note"))
        return f"{point} (no interval{'' if note == MISSING else f': {note}'})"
    return f"{point} [{bounds[0]}, {bounds[1]}]"


def _rows(value: Any) -> List[Dict[str, Any]]:
    """Normalise a table -- a DataFrame or a list of mappings -- into rows."""
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return [dict(row) for row in value]


def _table(rows: Sequence[Mapping[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    Written here rather than through ``DataFrame.to_markdown``, which needs a package this
    repository does not depend on. Every cell goes through :func:`_num` / :func:`_text`, so an
    absent measurement is :data:`MISSING` in a table exactly as it is in a sentence.

    Args:
        rows: The rows.
        columns: Column order. ``None`` takes the union of the rows' keys, first-seen order.

    Returns:
        The table, or a line saying the table is empty.
    """
    records = [dict(row) for row in rows]
    if not records:
        return f"_{MISSING}: no rows._"
    if columns is None:
        columns = []
        for record in records:
            for key in record:
                if key not in columns:
                    columns.append(key)

    def _cell(value: Any) -> str:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return _int(value)
        if isinstance(value, (float, np.floating)):
            return _num(value)
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return MISSING
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(str(name) for name in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in records:
        lines.append(
            "| " + " | ".join(_cell(record.get(name)) for name in columns) + " |"
        )
    return "\n".join(lines)


def _missing(what: str, reason: str = "") -> str:
    """The line a section prints when the run did not produce it.

    A section that renders nothing is indistinguishable from one whose numbers were all zero, and a
    reader has to be able to tell a software failure from a cohort too small to measure from a
    finding that came out inconclusive.

    Args:
        what: What is absent.
        reason: Why, when the record says.

    Returns:
        A blockquote naming the gap.
    """
    tail = f" {reason}" if reason else ""
    return f"> **{what}: {MISSING}.**{tail}"


# =============================================================================
# The written report
# =============================================================================
def reproduction_commands(protocol: Optional[Mapping[str, Any]]) -> List[str]:
    """The exact commands that reproduce this run, from its own persisted arguments.

    Args:
        protocol: The run's protocol record, or ``None``.

    Returns:
        One command per line, carrying the run's own ``--set`` overrides and its device. Empty
        when the protocol did not record its arguments, which is reported as a gap rather than
        filled with a plausible-looking command line.
    """
    run_args = dict((protocol or {}).get("run_args") or {})
    if not run_args:
        return []
    config_path = _text(run_args.get("config_path"))
    directory = (protocol or {}).get("run_directory")
    # The overrides are part of the run's identity, not decoration. Without them the first command
    # reproduces a different run, and the second is actively refused: re-entering a run directory
    # compares the resolved settings against the stored ones and raises on any difference.
    flags = "".join(f" --set {value}" for value in (run_args.get("set_overrides") or []))
    if run_args.get("device"):
        flags += f" --device {run_args['device']}"
    commands = [f"python -m {RUNNER_MODULE} --config {config_path} --stage all{flags}"]
    if directory:
        commands.append(
            f"python -m {RUNNER_MODULE} --config {config_path} --stage report "
            f"--run-dir {directory}{flags}"
        )
    return commands


def _provenance_section(record: Mapping[str, Any]) -> List[str]:
    """Checkpoint identity, statistics, software revision and exposure."""
    lines = ["## Provenance"]
    protocol = dict(record.get("protocol") or {})
    checkpoint = dict(record.get("checkpoint") or {})
    if not protocol and not checkpoint:
        # A blank line, or the table below is swallowed into the blockquote by every Markdown
        # renderer and the reader gets one long quoted paragraph instead of a table.
        lines.extend([
            _missing("run identity", "no protocol or checkpoint record was supplied."), ""
        ])

    settings = dict(protocol.get("settings") or {})
    paths = dict(settings.get("paths") or {})
    lines.append(_table([
        {"field": "run id", "value": _text(protocol.get("run_id"))},
        {"field": "created (UTC)", "value": _text(protocol.get("created_utc"))},
        {"field": "fold", "value": _text(protocol.get("fold"))},
        {"field": "seed", "value": _int(protocol.get("seed"))},
        {"field": "device", "value": _text(protocol.get("device"))},
        {"field": "checkpoint", "value": _text(
            checkpoint.get("checkpoint") or paths.get("checkpoint")
        )},
        {"field": "checkpoint digest", "value": _text(checkpoint.get("checkpoint_digest"))},
        {"field": "model class", "value": _text(checkpoint.get("model_class"))},
        {"field": "d_z", "value": _int(checkpoint.get("d_z"))},
        {"field": "statistics", "value": _text(paths.get("statistics"))},
        {"field": "software", "value": _text(protocol.get("software"))},
        {"field": "settings digest", "value": _text(protocol.get("settings_digest"))},
        # Section 4.1 asks for it because holdout and augmented builds partition differently, so
        # an unrecorded mode is a gap in what the held-out result generalizes to.
        {"field": "dataset build mode", "value": _text(record.get("dataset_build_mode"))},
    ], columns=["field", "value"]))

    exposure = dict(record.get("exposure") or {})
    lines.append("")
    lines.append("### Exposure of the held-out population")
    if not exposure:
        lines.append(_missing(
            "exposure",
            "no exposure record was supplied, so pretraining overlap is UNKNOWN -- which is not a "
            "statement that the splits are disjoint.",
        ))
        return lines
    lines.append(_table([
        {
            "population": name,
            "known": str(bool(dict(exposure.get(name) or {}).get("known"))),
            "n exposed": _int(dict(exposure.get(name) or {}).get("n_exposed")),
            "note": _text(dict(exposure.get(name) or {}).get("note")),
        }
        for name in ("pretraining", "selection")
    ], columns=["population", "known", "n exposed", "note"]))
    lines.append("")
    lines.append(
        f"- Statistics population: {_text(dict(exposure.get('statistics') or {}).get('population'))}"
    )
    lines.append(
        f"- Clean-holdout claim supported: "
        f"**{bool(exposure.get('clean_holdout_supported'))}**. Unknown provenance leaves this "
        f"false; it never becomes true by absence of evidence."
    )
    return lines


def _cohort_section(record: Mapping[str, Any]) -> List[str]:
    """Counts, coverage and every exclusion reason."""
    lines = ["## Cohort, coverage and exclusions"]
    cohort = dict(record.get("cohort") or {})
    if not cohort:
        lines.append(_missing("cohort", "no cohort record was supplied."))
        return lines

    lines.append("### Coverage by split")
    lines.append(_table(_rows(cohort.get("coverage"))))
    lines.append("")
    lines.append("### Exclusions")
    exclusions = dict(cohort.get("exclusions") or {})
    lines.append(_table(
        [{"reason": key, "n recordings": value} for key, value in sorted(exclusions.items())],
        columns=["reason", "n recordings"],
    ) if exclusions else _missing("exclusion counts"))
    lines.append("")
    lines.append("### Coverage contrast between outcome groups")
    lines.append(
        "A cloud separated mainly by ascertainment or by how much signal survived is not an "
        "outcome result. These rows are descriptive; nothing here adjusts anything."
    )
    lines.append("")
    lines.append(_table(_rows(cohort.get("coverage_contrast"))))
    return lines


def _selection_section(record: Mapping[str, Any]) -> List[str]:
    """What was fitted, what was selected, and against what."""
    lines = ["## Fitting and selection"]
    selection = dict(record.get("selection") or {})
    if not selection:
        lines.append(_missing("selection", "no baseline or adaptation record was supplied."))
        return lines

    baseline = dict(selection.get("baseline") or {})
    adaptation = dict(selection.get("adaptation") or {})
    lines.append("### Frozen baseline")
    lines.append(_table([
        {"field": key, "value": baseline.get(key)}
        for key in sorted(baseline) if not isinstance(baseline.get(key), (dict, list))
    ], columns=["field", "value"]) if baseline else _missing("baseline fit"))

    lines.append("")
    lines.append("### Adaptation")
    if not adaptation:
        lines.append(_missing("adaptation fit"))
        return lines
    epoch = adaptation.get("selected_epoch")
    lines.append(_table([
        {"field": key, "value": adaptation.get(key)}
        for key in sorted(adaptation) if not isinstance(adaptation.get(key), (dict, list))
    ], columns=["field", "value"]))
    lines.append("")
    if epoch is not None and int(epoch) == 0:
        lines.append(
            "**Selection retained the frozen model (epoch 0).** No adapted candidate improved "
            "eligible validation AUROC, so the pretrained mean-output layers are what this run "
            "reports. That is a result, not a failure to select."
        )
    elif epoch is not None:
        lines.append(
            f"**Selection chose epoch {int(epoch)}**, the highest eligible validation AUROC among "
            f"candidates that passed the preservation gates."
        )
    return lines


def _preservation_section(record: Mapping[str, Any]) -> List[str]:
    """The declared gates, what they measured, and what they only warned about."""
    lines = ["## Preservation"]
    preservation = dict(record.get("preservation") or {})
    if not preservation:
        lines.append(_missing("preservation", "no gate record was supplied."))
        return lines

    gate = dict(preservation.get("gate") or {})
    if gate:
        lines.append(_table([
            {"field": key, "value": gate.get(key)}
            for key in (
                "passed", "rule",
                "forecast_mse_max_increase", "baseline_mse_full", "candidate_mse_full",
                "mse_full_increase",
                "saturation_max_increase_pp", "baseline_delta_mu_sat_pp",
                "candidate_delta_mu_sat_pp", "delta_mu_sat_increase_pp",
                "baseline_mse_full_healthy", "candidate_mse_full_healthy",
                "mse_full_healthy_increase",
                "n_latent_coordinates_varying", "n_recordings", "n_retained_anchors",
                "support_digest",
            ) if key in gate
        ], columns=["field", "value"]))
        lines.append("")
        for reason in list(gate.get("reasons") or []):
            lines.append(f"- **Gate failure:** {reason}")
        for warning in list(gate.get("warnings") or []):
            lines.append(f"- *Reported, not gated:* {warning}")
        lines.append("")
        lines.append(
            "Both tolerances are declared engineering thresholds, fixed before training. Neither "
            "is clinically validated, and a fixed decoder does not preserve forecasts on its own "
            "when its latent input moves."
        )
    else:
        lines.append(_missing("gate decision"))

    lines.append("")
    lines.append("### Matched-policy Monte Carlo NLL and KL")
    nll = _rows(preservation.get("nll"))
    lines.append(_table(nll) if nll else _missing("MC NLL / KL diagnostics"))
    convergence = dict(preservation.get("convergence") or {})
    if convergence:
        lines.append("")
        lines.append(_table([
            {"field": key, "value": value} for key, value in sorted(convergence.items())
        ], columns=["field", "value"]))
    return lines


def _metrics_section(record: Mapping[str, Any]) -> List[str]:
    """The primary held-out table: one observation per recording, paired intervals."""
    lines = ["## Held-out discrimination"]
    metrics = dict(record.get("metrics") or {})
    if not metrics:
        lines.append(_missing(
            "held-out metrics",
            "the evaluation stage produced no record, so this run makes no discrimination claim.",
        ))
        return lines

    models = dict(metrics.get("models") or {})
    if models:
        lines.append(_table([
            {
                "model": name,
                "n recordings": measured.get("n_recordings"),
                "n healthy": measured.get("n_healthy"),
                "n adverse": measured.get("n_adverse"),
                "prevalence": measured.get("prevalence"),
                "AUROC": measured.get("auroc"),
                "AP": measured.get("average_precision"),
                "chance AP": measured.get("chance_average_precision"),
                "balanced acc": measured.get("balanced_accuracy"),
                "threshold": measured.get("threshold"),
            }
            for name, measured in models.items()
        ]))
    else:
        lines.append(_missing("per-model metrics"))

    bootstrap = dict(metrics.get("bootstrap") or {})
    lines.append("")
    lines.append("### Paired differences")
    if not bootstrap:
        lines.append(_missing("paired bootstrap"))
    elif bootstrap.get("error"):
        # A cohort too small to resample is a result about this fold, and the run wrote down why.
        # Falling through to the measured branch below would render that reason as six separate
        # "not measured" placeholders -- the one thing the record actually knows, discarded.
        lines.append(
            _missing(
                "paired intervals",
                f"{bootstrap['error']} ({_int(bootstrap.get('n_recordings'))} recording(s))",
            )
        )
    else:
        paired = dict(bootstrap.get("paired") or {})
        lines.append(_table([
            {
                "comparison": comparison,
                **{
                    metric: _ci(dict(values.get(metric) or {}))
                    for metric in sorted(values)
                },
            }
            for comparison, values in paired.items()
        ]) if paired else _missing("paired intervals"))
        lines.append("")
        lines.append(
            f"- {_text(bootstrap.get('method'))}; {_int(bootstrap.get('resamples'))} draw(s) over "
            f"{_int(bootstrap.get('n_units'))} {_text(bootstrap.get('grouping'))}(s), "
            f"{_int(bootstrap.get('n_undefined_draws'))} undefined draw(s) excluded."
        )
        lines.append(f"- {_text(bootstrap.get('note'))}")
        if bootstrap.get("grouping_disclosure"):
            lines.append(f"- **{_text(bootstrap.get('grouping_disclosure'))}**")

    centroid = dict(metrics.get("nearest_centroid") or {})
    lines.append("")
    lines.append("### Nearest-centroid geometry check")
    lines.append(
        "Head-independent, in the same fixed standardized latent space: linear discrimination can "
        "improve while the classes' centres do not move, and that combination is stated rather "
        "than smoothed over."
    )
    lines.append("")
    lines.append(_table([
        {"model": name, **{key: value for key, value in dict(measured).items()}}
        for name, measured in centroid.items()
    ]) if centroid else _missing("nearest-centroid metrics"))

    # The heading is unconditional: a section that renders nothing where a measurement was
    # expected reads as "there was nothing to say", which is the one thing this report must not
    # let a reader conclude. Absent records become a named gap instead, as everywhere else here.
    geometry = dict(record.get("geometry") or {})
    lines.append("")
    lines.append("### Latent movement and spread")
    movement = dict(geometry.get("movement") or {})
    lines.append(_table([
        {"field": key, "value": value}
        for key, value in sorted(movement.items())
        if not isinstance(value, (list, dict))
    ], columns=["field", "value"]) if movement else _missing("latent movement"))
    covariance = dict(geometry.get("covariance") or {})
    lines.append("")
    lines.append(_table([
        {
            "model": name,
            "n recordings": dict(values).get("n_recordings"),
            "total variance": dict(values).get("total_variance"),
            "effective rank": dict(values).get("effective_rank"),
            "leading share": dict(values).get("leading_share"),
        }
        for name, values in covariance.items()
    ]) if covariance else _missing("latent covariance and effective rank"))
    return lines


#: The classification columns the held-out table prints, in reporting order. The ranking metrics
#: first, then the threshold-dependent rates, then the cells they are computed from.
CLASSIFICATION_COLUMNS: Tuple[str, ...] = (
    "model", "auroc", "average_precision", "balanced_accuracy", "f1",
    "sensitivity", "specificity", "precision", "npv", "accuracy",
    "tp", "fp", "fn", "tn", "threshold", "prevalence",
    "n_recordings", "n_healthy", "n_adverse",
)

#: The per-bin columns, with the identity of the cell first and the counts last, because the counts
#: are what a per-bin number has to be read against.
BIN_COLUMNS: Tuple[str, ...] = (
    "model", "time_bin", "time_bin_label", "supervised_window", "estimable",
    "auroc", "auroc_lo", "auroc_hi", "average_precision", "f1", "balanced_accuracy",
    "sensitivity", "specificity", "tp", "fp", "fn", "tn",
    "n_recordings", "n_healthy", "n_adverse",
)


def _classification_section(record: Mapping[str, Any]) -> List[str]:
    """The held-out confusion counts and every rate derived from them."""
    measured = dict(dict(record.get("metrics") or {}).get("classification") or {})
    lines = ["## Held-out classification"]
    if not measured:
        lines.append(_missing(
            "the classification table",
            "the report stage did not produce one; it is derived from "
            "`per_recording_test.parquet` and the locked thresholds.",
        ))
        return lines
    rows = [{"model": name, **dict(values)} for name, values in measured.items()]
    lines.append(_table(rows, CLASSIFICATION_COLUMNS))
    lines.extend([
        "",
        "Counts are **recordings**, one per held-out recording, at each model's own threshold. "
        "That threshold was chosen on validation to maximise balanced accuracy and is never "
        "re-chosen here; AUROC and average precision do not depend on it, and are the two numbers "
        "to compare between models whose logit scales are not comparable.",
        "",
        "`precision`, `npv` and `accuracy` carry no interval: the paired bootstrap resamples the "
        "metrics named in `evaluate.METRIC_NAMES`, and these three are reported as point estimates "
        "rather than given an interval the run did not compute.",
    ])
    return lines


def _time_section(record: Mapping[str, Any]) -> List[str]:
    """Discrimination bin by bin, with what each bin rests on."""
    rows = _rows(dict(record.get("metrics") or {}).get("per_bin"))
    lines = ["## Discrimination over time before delivery"]
    if not rows:
        lines.append(_missing(
            "the per-bin table",
            "no trajectory bin produced a scored cell.",
        ))
        return lines
    lines.append(_table(rows, BIN_COLUMNS))
    estimable = sum(1 for row in rows if bool(row.get("estimable")))
    lines.extend([
        "",
        f"{estimable} of {len(rows)} (model, bin) cells carried both outcome groups and could be "
        f"measured; the rest report their counts and no metric rather than a number standing in "
        f"for one. A cell marked `estimable` false still carries its four confusion cells and the "
        f"rates a single class leaves defined -- specificity and NPV on a bin with no adverse "
        f"recording, and precision and F1 at zero under the never-fires convention -- because "
        f"those are measured facts about that bin. It is *discrimination between the two groups* "
        f"that is undefined there, which is why the figures draw a gap rather than a point.",
        "",
        "Three things this table is not. The head is the **frozen final-hour classifier**, so "
        "every row with `supervised_window` false is that head applied outside the window its loss "
        "was defined on. The **threshold came from validation**, on final-hour bags, so an earlier "
        "bin's counts are that rule applied where it was not tuned. And the **cohort is whoever "
        "was observed in that bin** -- a recording contributes only where it has retained anchors "
        "-- so a metric moving towards delivery may be a trend or a change in who was still being "
        "recorded, and `n_recordings` beside it is the only thing that separates the two.",
    ])
    return lines


def _controls_section(record: Mapping[str, Any]) -> List[str]:
    """The shuffled-label control, the prior probe, and what neither establishes."""
    lines = ["## Controls"]
    controls = dict(record.get("controls") or {})
    if not controls:
        lines.append(_missing("controls", "no control record was supplied."))
        return lines

    measured = dict(controls.get("metrics") or {})
    lines.append(_table([
        {"control": name, **{key: value for key, value in dict(values).items()}}
        for name, values in measured.items()
    ]) if measured else _missing("control metrics"))

    disclosure = dict(controls.get("disclosure") or {})
    lines.append("")
    if disclosure:
        lines.append(f"- {_text(disclosure.get('shuffled_label_note'))}")
        lines.append(f"- {_text(disclosure.get('prior_probe_note'))}")
        lines.append(
            f"- Permutation p-value: **{bool(disclosure.get('permutation_p_value'))}**. "
            f"Combined-branch claim supported: "
            f"**{bool(disclosure.get('combined_branch_claim_supported'))}**."
        )
    else:
        lines.append(_missing("control disclosure"))
    return lines


def _temporal_section(record: Mapping[str, Any]) -> List[str]:
    """The paired early/late change, per model, with the sentence it supports."""
    # From the run's own settings where they were recorded; a geometry-free heading otherwise,
    # rather than a default invented here that a moved window would silently contradict.
    windows = dict(
        dict(dict(record.get("protocol") or {}).get("settings") or {}).get("windows") or {}
    )
    hours = windows.get("analysis_hours") or windows.get("preservation_hours")
    lines = [
        f"## Change over the last {float(hours):g} hours" if hours is not None
        else "## Change before delivery"
    ]
    temporal = dict(record.get("temporal") or {})
    if not temporal:
        lines.append(_missing("temporal analysis"))
        return lines

    rows: List[Dict[str, Any]] = []
    for name, contrast in temporal.items():
        entry = dict(contrast)
        groups = dict(entry.get("groups") or {})
        rows.append({
            "model": name,
            "grouped by": _text(entry.get("group_column")),
            "n paired": entry.get("n_paired_recordings"),
            "adverse delta": _ci(dict(groups.get("adverse") or {})),
            "n adverse": dict(groups.get("adverse") or {}).get("n_recordings"),
            "healthy delta": _ci(dict(groups.get("healthy") or {})),
            "n healthy": dict(groups.get("healthy") or {}).get("n_recordings"),
            "difference": _ci(dict(entry.get("difference") or {})),
        })
    lines.append(_table(rows))
    first = dict(next(iter(temporal.values()), {}) or {})
    lines.append("")
    lines.append(f"- {_text(first.get('interpretation'))}")
    lines.append(f"- {_text(first.get('limitation'))}")
    return lines


def _subgroup_section(record: Mapping[str, Any]) -> List[str]:
    """Prespecified strata, reported together and never promoted afterwards."""
    lines = ["## Prespecified subgroups"]
    rows = _rows(record.get("subgroups"))
    if not rows:
        lines.append(_missing("subgroup table"))
        return lines
    lines.append(
        "Training is binary and the headline stays binary. These rows are produced together, in a "
        "fixed order, for every model at once -- so that a reader can see whether a result rests "
        "on one stratum, not so the best one can be chosen afterwards. Nothing here is refitted."
    )
    lines.append("")
    lines.append(_table(rows))
    return lines


def _figures_section(record: Mapping[str, Any]) -> List[str]:
    """The figures this run wrote, each with the caption it carries."""
    lines = ["## Figures"]
    figures_written = dict(record.get("figures") or {})
    if not figures_written:
        lines.append(_missing("figures", "no figure was rendered for this run."))
        return lines
    for name, path in figures_written.items():
        lines.append(f"- `{path}`")
        caption = CAPTIONS.get(str(name))
        if caption:
            lines.append(f"  - {caption}")
    return lines


def _limitations_section(record: Mapping[str, Any]) -> List[str]:
    """What this design cannot establish, whatever the numbers above say."""
    exposure = dict(record.get("exposure") or {})
    bootstrap = dict(dict(record.get("metrics") or {}).get("bootstrap") or {})
    fixed = [
        "One fold, one checkpoint, one seed. Every interval here describes uncertainty "
        "conditional on that fold and that fitting seed, not variation across training runs.",
        "Supervision is one final-hour summary per recording. The labels are outcomes, not "
        "time-resolved physiological states: no result here assigns a state or an injury onset to "
        "any individual time point, and bins outside the supervised hour are applications of the "
        "head outside the window it was fitted on.",
        "The adaptation touches the posterior mean-output layers only. A negative result is a "
        "limit of that small adaptation, not evidence that the architecture cannot learn the "
        "distinction.",
        "The latent support inherits the checkpoint's forecast-availability exclusions, so it is "
        "not every theoretically inferable pre-delivery state.",
        "Class-balanced training makes the logits association scores. Their sigmoid is not a "
        "calibrated clinical risk.",
        "The stored class codes are categories, not an ordinal severity scale; the acidosis and "
        "HIE contrasts are descriptive and were not fitted separately.",
        "Eligibility requires late coverage -- a minimum number of contributing segments inside "
        "the supervised window and a retained anchor near delivery (counts by reason are in the "
        "Cohort section). Every metric, band and figure here describes that coverage-selected "
        "subset, which is not a random sample of the cohort; pairing and stratification do not "
        "remove that selection.",
    ]
    conditional: List[str] = []
    if not exposure:
        # The guard used to require a record to exist, so the case with no provenance at all --
        # the least certain one there is -- was the single case that produced no caveat.
        conditional.append(
            "No exposure record was produced for this run, so pretraining and checkpoint-selection "
            "overlap with the held-out recordings is UNKNOWN. Unknown is not disjoint: treat this "
            "as exploratory reuse rather than a pristine final test."
        )
    elif not exposure.get("clean_holdout_supported"):
        conditional.append(
            "Pretraining/selection exposure of the held-out recordings is not established, so "
            "this is exploratory reuse of that population rather than a pristine final test."
        )
    if bootstrap.get("grouping_disclosure"):
        conditional.append(str(bootstrap["grouping_disclosure"]) + ".")
    extra = [str(line) for line in (record.get("limitations") or [])]
    return ["## Limitations"] + [f"- {line}" for line in fixed + conditional + extra]


def _reproduction_section(record: Mapping[str, Any]) -> List[str]:
    """The commands that regenerate this run."""
    lines = ["## Reproduction"]
    commands = list(record.get("reproduction") or []) or reproduction_commands(
        record.get("protocol")
    )
    if not commands:
        lines.append(_missing(
            "reproduction commands", "the run's arguments were not recorded in its protocol."
        ))
        return lines
    lines.append("```bash")
    lines.extend(str(command) for command in commands)
    lines.append("```")
    return lines


def build_report(record: Mapping[str, Any]) -> str:
    """Assemble the measured report from a finished run's records.

    Every number in the output comes out of ``record``. Nothing is defaulted to a plausible value,
    no example result is filled in, and a section whose record is absent says so in place of its
    table -- so a reader can tell a stage that never ran from a cohort too small to measure from a
    finding that came out inconclusive.

    The record's sections, all optional:

    ``protocol``
        The run's protocol record, from :func:`~latent_pilot.config.read_protocol`.
    ``checkpoint``
        The pretrained checkpoint's identity: path, digest, class, geometry.
    ``exposure``
        :func:`~latent_pilot.data.exposure_record`.
    ``cohort``
        ``{'coverage': ..., 'exclusions': ..., 'coverage_contrast': ...}``.
    ``selection``
        ``{'baseline': BaselineFit.record, 'adaptation': AdaptationFit.record}``.
    ``preservation``
        ``{'gate': GateResult.record, 'nll': ..., 'convergence': ...}``.
    ``metrics``
        ``{'models': {name: recording_metrics}, 'bootstrap': paired_bootstrap record,
        'nearest_centroid': {name: recording_metrics}}``, and optionally ``'classification'``
        (``{name: recording_metrics}`` on the held-out final-hour bags) and ``'per_bin'`` (the rows
        of :func:`~latent_pilot.analyze.bin_classification`). Those two are added by the report
        stage rather than the evaluation stage, so a run finished before they existed gains them
        the next time its report is regenerated.
    ``geometry``
        ``{'movement': ..., 'covariance': {name: ...}}``.
    ``controls``
        ``{'metrics': {name: recording_metrics}, 'disclosure': control_disclosure}``.
    ``temporal``
        ``{model name: paired_contrast record}``.
    ``subgroups``
        The subgroup table.
    ``figures``
        ``{figure name: path}``.
    ``limitations``
        Extra limitation lines beyond the fixed ones this design always carries.
    ``reproduction``
        Explicit commands; derived from the protocol when absent.

    Args:
        record: The run's records.

    Returns:
        The report, as Markdown.
    """
    protocol = dict(record.get("protocol") or {})
    header = [
        "# Latent-class fine-tuning pilot -- measured report",
        "",
        f"Run `{_text(protocol.get('run_id'))}`, fold `{_text(protocol.get('fold'))}`, seed "
        f"`{_int(protocol.get('seed'))}`, written {_text(protocol.get('created_utc'))}.",
        "",
        "One prespecified fold, one pretrained checkpoint, one seed. This report states what was "
        "measured and what was not. Every value marked "
        f"*{MISSING}* is one this run did not establish, not one that came out at zero.",
    ]
    sections: List[List[str]] = [
        header,
        _provenance_section(record),
        _cohort_section(record),
        _selection_section(record),
        _preservation_section(record),
        _metrics_section(record),
        _classification_section(record),
        _time_section(record),
        _controls_section(record),
        _temporal_section(record),
        _subgroup_section(record),
        _figures_section(record),
        _limitations_section(record),
        _reproduction_section(record),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_report(record: Mapping[str, Any], directory: Any) -> Path:
    """Write the measured report into a run directory.

    Args:
        record: The run's records, as :func:`build_report` documents them.
        directory: The run directory. Created if absent.

    Returns:
        The written path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / REPORT_FILENAME
    target.write_text(build_report(record), encoding="utf-8")
    logger.info(f"report written: {target}")
    return target


__all__ = [
    "CAPTIONS",
    "DEFAULT_N_ARROWS",
    "DEFAULT_N_TRACES",
    "FIGURE_DIRNAME",
    "FIGURE_CONFUSION",
    "FIGURE_COUNTS_TIME",
    "FIGURE_COVERAGE_SPACE",
    "FIGURE_LATENT_SPACE",
    "FIGURE_METRICS_TIME",
    "FIGURE_ROC_BINS",
    "FIGURE_ROC_PR",
    "FIGURE_SUPERVISED_AXIS",
    "FIGURE_TRAJECTORIES",
    "MISSING",
    "PLOT_CLASSES",
    "REPORT_FILENAME",
    "RUNNER_MODULE",
    "axis_label",
    "build_report",
    "class_palette",
    "configure_figures",
    "coverage_groupings",
    "figure_confusion",
    "figure_counts_vs_time",
    "figure_coverage_space",
    "figure_latent_space",
    "figure_metrics_vs_time",
    "figure_roc_by_bin",
    "figure_roc_pr",
    "figure_supervised_axis",
    "figure_trajectories",
    "reproduction_commands",
    "score_scale_note",
    "select_traces",
    "signed_bin_hours",
    "variance_caption",
    "with_class_names",
    "write_report",
]
