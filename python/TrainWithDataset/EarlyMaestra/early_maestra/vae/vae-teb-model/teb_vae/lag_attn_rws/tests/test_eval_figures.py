r"""The three properties a figure layer must have before any analysis draws through it.

**An empty panel must be legible, not fatal.** An analysis that legitimately found nothing -- a
fully masked split, a metric undefined for this checkpoint -- reaches its figure at the end of a
run that has already cost hours. Raising there loses everything; drawing an unlabelled empty
frame produces something that reads as a bug. So it draws the note.

**Importing must not restyle anything.** ``apply_publication_style`` mutates global ``rcParams``,
so an import-time call silently restyles every other figure produced in the same process --
including one a test is asserting on, and including the training callback's if an evaluation is
ever run in-process beside it. Checked in a subprocess, because this session has imported the
module long before the test runs and an in-process check would pass no matter what.

**A rendered figure must be closed.** matplotlib holds every unclosed figure alive in a global
registry. A production pass draws one per analysis per grouping axis, so a leak here is measured
in hundreds of megabytes rather than in tidiness.

The fourth test is the one property that is this package's rather than the shared layer's: a lag
axis is drawn in *compensated* seconds, and the label naming that quantity is the model's own.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_rws.eval import figures_seam

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_the_seam_binds_the_shared_panels_rather_than_copies() -> None:
    """Identity: a fork would pass every behavioural test below and still drift from the
    training figures it must stay colour-consistent with."""
    assert figures_seam.render_figure is shared_figures.render_figure
    assert figures_seam.violin_panel is shared_figures.violin_panel
    assert figures_seam.grouped_violin_figure is shared_figures.grouped_violin_figure
    assert figures_seam.EMPTY_NOTE == shared_figures.EMPTY_NOTE


def test_an_all_nan_series_draws_the_empty_note_rather_than_raising() -> None:
    figure, axes = figures_seam.new_figure(1)
    try:
        drawn = figures_seam.histogram_panel(
            axes[0, 0], np.full(32, np.nan), title="pred_gap"
        )
        texts = [text.get_text() for text in axes[0, 0].texts]
    finally:
        shared_figures.plt.close(figure)

    assert drawn == 0
    assert figures_seam.EMPTY_NOTE in texts


def test_an_empty_group_keeps_its_slot_so_the_labels_stay_aligned() -> None:
    """Dropping the empty group would shift every label after it onto the wrong violin."""
    figure, axes = figures_seam.new_figure(1)
    try:
        populated = figures_seam.violin_panel(
            axes[0, 0],
            {"healthy": [1.0, 2.0, 3.0], "acidosis": [], "hie": [4.0, 5.0, 6.0]},
        )
        labels = [text.get_text() for text in axes[0, 0].get_xticklabels()]
    finally:
        shared_figures.plt.close(figure)

    assert populated == 2
    assert labels == ["healthy", "acidosis", "hie"]


def test_the_violin_interior_draws_the_quartiles_and_the_adjacent_values() -> None:
    r"""The mark inside a violin is what a reader takes the middle half off, so it has to be the
    quartiles rather than anything that merely looks like them.

    The sample carries one extreme value, and that is what makes the test non-vacuous: it
    separates the two whisker conventions. An implementation drawing the **range** reaches $100$;
    Tukey's adjacent value stops at the furthest observation inside the fence, which is $9$. The
    outlier is still on the page either way -- matplotlib evaluates the violin's kernel density
    between the data's own extremes, so it is the body's tail.
    """
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]

    figure, axes = figures_seam.new_figure(1)
    try:
        figures_seam.violin_panel(axes[0, 0], {"healthy": values})
        spans = sorted(
            (round(float(segment[0][1]), 6), round(float(segment[1][1]), 6))
            for artist in axes[0, 0].collections
            if isinstance(artist, LineCollection)
            for segment in artist.get_segments()
        )
        medians = [
            float(line.get_ydata()[0])
            for line in axes[0, 0].lines
            if line.get_marker() == "o"
        ]
    finally:
        shared_figures.plt.close(figure)

    # The whisker first, then the inter-quartile bar: $Q_1 = 3.25$, $Q_3 = 7.75$ on this sample.
    assert spans == [(1.0, 9.0), (3.25, 7.75)]
    assert medians == [5.5]


def test_render_figure_writes_the_file_and_leaves_no_open_figure(tmp_path) -> None:
    shared_figures.plt.close("all")
    figure, axes = figures_seam.new_figure(1)
    figures_seam.histogram_panel(axes[0, 0], np.linspace(0.0, 1.0, 32), title="pred_gap")

    path = figures_seam.render_figure(figure, tmp_path / "pred_gap")

    assert Path(path).is_file() and Path(path).stat().st_size > 0
    assert shared_figures.plt.get_fignums() == []


def test_the_lag_axis_label_is_the_models_own_compensated_one() -> None:
    r"""There are two seconds figures for one lag index and they are different quantities. The
    sensor figure adds back the $20$ s the preprocessing already removed and exists only to map a
    finding onto the original files; a figure that drew it under this name would double-count a
    deliberate correction."""
    from teb_vae.lag_attn.nets import lag_report

    assert figures_seam.COMPENSATED_LAG_AXIS_LABEL is lag_report.COMPENSATED_LAG_AXIS_LABEL
    assert "compensated" in figures_seam.COMPENSATED_LAG_AXIS_LABEL


def test_importing_the_seam_does_not_restyle_the_process() -> None:
    """In a subprocess: this session imported the module long ago, so an in-process comparison
    would pass whatever the module does at import time."""
    source = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "before = dict(plt.rcParams)\n"
        "from teb_vae.lag_attn_rws.eval import figures_seam\n"
        "moved = sorted(k for k, v in plt.rcParams.items() if before.get(k) != v)\n"
        "print(','.join(moved))\n"
        # And the styling is available, so it is opt-in rather than absent.
        "figures_seam.configure_figure_style()\n"
        "assert sorted(k for k, v in plt.rcParams.items() if before.get(k) != v)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", (
        f"importing figures_seam moved rcParams {completed.stdout.strip()}; styling must be an "
        f"explicit call at run start, not an import side effect"
    )


def test_a_grouped_violin_figure_draws_one_body_per_populated_group() -> None:
    values = {"pred_gap": {"healthy": np.array([1.0, 2.0, 3.0]),
                           "acidosis": np.array([4.0, 5.0, 6.0])}}

    figure, axes = figures_seam.grouped_violin_figure(values, ["healthy", "acidosis"])
    try:
        bodies = [
            artist for artist in axes[0, 0].collections if isinstance(artist, PolyCollection)
        ]
    finally:
        shared_figures.plt.close(figure)

    assert len(bodies) == 2


# =============================================================================
# Every emitted figure is documented
# =============================================================================
#: Every PDF the shipped analyses can emit, as ``(module, attribute)``. Read off the analysis
#: modules rather than listed as strings, so a renamed figure fails here rather than quietly
#: leaving its guide entry pointing at a file nothing writes.
_FIGURE_CONSTANTS = (
    ("forecast", ("BASELINE_FIGURE", "ANCHOR_FIGURE", "OVERLAY_FIGURE", "HORIZON_FIGURE")),
    ("coupling", ("DISTRIBUTION_FIGURE", "PERCENT_FIGURE")),
    ("latent", ("SPECTRUM_FIGURE",)),
    ("lag_kl", ("PROFILE_FIGURE",)),
    ("attention", ("PROFILE_FIGURE", "HEATMAP_FIGURE")),
    ("calibration", ("PIT_FIGURE", "LOGVAR_FIGURE")),
    (
        "coherence",
        (
            "LEAD_TIME_FIGURE",
            "SPECTRUM_FIGURE",
            "BANDS_FIGURE",
            "DECOMPOSITION_FIGURE",
            "SOURCE_FIGURE",
            "SEAM_FIGURE",
        ),
    ),
    ("distributions", ("CLASS_FIGURE", "SUBGROUP_FIGURE")),
    ("trajectory", ("PROFILE_FIGURE",)),
    ("time_to_delivery", ("TRAJECTORY_FIGURE", "WINDOWS_FIGURE")),
    ("second_stage", ("TRAJECTORY_FIGURE", "WINDOWS_FIGURE")),
    ("events", ("DECELERATION_FIGURE", "TRIGGERED_FIGURE", "CONDITIONED_FIGURE")),
    ("sufficiency", ("SUFFICIENCY_FIGURE",)),
    ("cross_subgroup", ("HEATMAP_FIGURE",)),
)

#: The document every one of them must appear in.
FIGURE_GUIDE = Path(__file__).resolve().parents[1] / "eval" / "FIGURE_GUIDE.md"


def _emitted_figures() -> dict:
    """Return ``{analysis: [figure filename, ...]}`` from the analysis modules themselves.

    The constants are **stems** -- the extension is the run's, from ``eval_config.figure_format``
    -- so the default format is appended here. That is the filename the figure guide documents,
    and comparing the two only means something if both carry it: a bare stem is a substring of
    every name built from it, so the has-an-entry direction would pass on a guide that documented
    the wrong extension.
    """
    import importlib

    from teb_vae.lag_attn.eval.figures import DEFAULT_FIGURE_FORMAT

    found = {}
    for stem, attributes in _FIGURE_CONSTANTS:
        module = importlib.import_module(f"teb_vae.lag_attn_rws.eval.analyses.{stem}")
        found[stem] = [
            f"{getattr(module, name)}.{DEFAULT_FIGURE_FORMAT}" for name in attributes
        ]
    return found


def test_every_emitted_figure_has_an_entry_in_the_figure_guide() -> None:
    """A figure nobody documented is a figure whose axes a reader has to reverse-engineer -- and
    the compensated-lag convention is exactly the kind of thing that gets reverse-engineered
    wrong. Bound to the *constants*, so renaming a file without moving its entry fails."""
    guide = FIGURE_GUIDE.read_text(encoding="utf-8")

    missing = [
        f"{analysis}/{name}"
        for analysis, names in _emitted_figures().items()
        for name in names
        if f"{analysis}/{name}" not in guide
    ]

    assert missing == [], f"{FIGURE_GUIDE.name} has no entry for {missing}"


def test_the_figure_guide_documents_nothing_that_is_not_emitted() -> None:
    """The other direction: an entry for a figure no analysis writes is a promise the run does not
    keep, and it outlives the analysis that was deleted."""
    import re

    guide = FIGURE_GUIDE.read_text(encoding="utf-8")
    documented = set(re.findall(r"`([a-z_]+/[a-z_]+\.pdf)`", guide))
    emitted = {
        f"{analysis}/{name}"
        for analysis, names in _emitted_figures().items()
        for name in names
    }

    assert documented - emitted == set()


# =============================================================================
# The clinical figures
#
# Two properties they share and neither is visual: the cohort colours come from this package's one
# clinical palette, so a class is the same colour on every figure of this evaluation; and a split
# with one cohort emits nothing rather than a single violin inviting a comparison there is nothing
# to compare against.
#
# ``test_eval_cohort_presentation.py`` is where that palette and the cohort order it travels with
# are pinned. What is asserted here is only the seam's part in it.
# =============================================================================
def test_a_cohort_keeps_the_same_colour_across_every_figure_that_draws_it() -> None:
    """Asserted against the mapping rather than eyeballed. A figure whose classes are coloured by
    whatever order they arrived in cannot be compared with any other figure in the run, and the
    failure is invisible until two figures are put side by side."""
    from teb_vae.lag_attn.eval import labels as shared_labels

    names = [shared_labels.CLASS_NAMES[code] for code in sorted(shared_labels.CLASS_NAMES)]
    colours = figures_seam.group_colors(names)

    assert set(colours) == set(names)
    # One mapping, and it is order-independent: the same cohort asked for in another order, or
    # alone, must come back the same colour.
    assert figures_seam.group_colors(list(reversed(names))) == colours
    assert figures_seam.group_colors([names[1]])[names[1]] == colours[names[1]]


def test_the_seam_overrides_the_shared_palette_rather_than_binding_it() -> None:
    """The one place this seam deliberately does *not* bind the sibling's function.

    ``utils.style.CLASS_COLORS_DEFAULT`` paints ``healthy`` blue and is shared with the
    ``lag_attn`` sibling and with ``model/transformer_experiment``, so it is overridden here rather
    than repainted there. A future edit reverting to the bare binding would restore blue on every
    cohort figure in this evaluation, and no other test would say so.
    """
    from teb_vae.lag_attn.eval import figures as shared

    assert figures_seam.group_colors is not shared.group_colors
    assert figures_seam.group_colors(["healthy"]) != shared.group_colors(["healthy"])


def test_the_subgroup_heatmap_reports_the_absence_of_a_survivor_rather_than_drawing_nothing() -> None:
    """An empty lower panel is a *result* -- no metric survived Holm -- and it says so in its
    title rather than arriving as a blank figure that reads as a plotting failure."""
    from teb_vae.lag_attn_rws.eval.analyses import cross_subgroup

    record = {
        "group_column": "subgroup",
        "alpha": 0.05,
        "omnibus": [],
        "pairwise": {},
        "significant_metrics": [],
        "n_metrics_tested": 0,
        "missing_sources": [],
        "method": "",
    }

    figure = cross_subgroup.build_heatmap_figure(record)
    try:
        notes = [text.get_text() for text in figure.axes[0].texts]
        lower_title = figure.axes[1].get_title()
    finally:
        shared_figures.plt.close(figure)

    assert notes == [figures_seam.EMPTY_NOTE]
    assert "no metric survived Holm" in lower_title
