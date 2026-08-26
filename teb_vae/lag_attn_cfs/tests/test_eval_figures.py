r"""The properties a figure layer must have before any analysis draws through it.

**An empty panel must be legible, not fatal.** An analysis that legitimately found nothing -- a
fully masked split, a metric undefined for this checkpoint -- reaches its figure at the end of a
run that has already cost hours. Raising there loses everything; drawing an unlabelled empty frame
produces something that reads as a bug. So it draws the note.

**Importing must not restyle anything.** ``apply_publication_style`` mutates global ``rcParams``,
so an import-time call silently restyles every other figure produced in the same process --
including one a test is asserting on, and including the training callback's if an evaluation is
ever run in-process beside it. Checked in a subprocess, because this session has imported the
module long before the test runs and an in-process check would pass no matter what.

**A rendered figure must be closed.** matplotlib holds every unclosed figure alive in a global
registry. A production pass draws one per analysis per grouping axis, so a leak here is measured in
hundreds of megabytes rather than in tidiness.

**The palette is a table rather than an assignment pass**, which is what makes it
order-independent: a cohort asked for alone, among others, or in another order comes back the same
colour, so two figures of overlapping cohorts can be put side by side. It is deliberately not
``utils.style.CLASS_COLORS_DEFAULT`` -- that table is shared with two other projects, so repainting
it there to satisfy this convention would restyle their figures. The cost is stated rather than
hidden and is asserted below: an evaluation figure of a cohort is *not* the same colour as a
training-callback figure of that cohort.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_cfs.eval import figures_seam
from teb_vae.lag_attn_cfs.eval._reuse import labels

from .conftest import _REPO_ROOT

#: The canonical subgroup order, restated so this file states the contract rather than the code.
EXPECTED_SUBGROUP_ORDER = [
    "healthy_no_bg_no_cs",
    "healthy_no_bg_cs",
    "healthy_bg_no_cs",
    "healthy_bg_cs",
    "acidosis_no_cs",
    "acidosis_cs",
    "hie_no_cs",
    "hie_cs",
]


# =================================================================================================
# The seam binds rather than forks
# =================================================================================================
def test_the_seam_binds_the_shared_panels_rather_than_copies() -> None:
    """Identity: a fork would pass every behavioural test below and still drift from the training
    figures it must stay colour-consistent with."""
    assert figures_seam.render_figure is shared_figures.render_figure
    assert figures_seam.violin_panel is shared_figures.violin_panel
    assert figures_seam.grouped_violin_figure is shared_figures.grouped_violin_figure
    assert figures_seam.EMPTY_NOTE == shared_figures.EMPTY_NOTE


def test_an_all_nan_series_draws_the_empty_note_rather_than_raising() -> None:
    figure, axes = figures_seam.new_figure(1)
    try:
        drawn = figures_seam.histogram_panel(axes[0, 0], np.full(32, np.nan), title="pred_gap")
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
        drawn = [text.get_text() for text in axes[0, 0].get_xticklabels()]
    finally:
        shared_figures.plt.close(figure)

    assert populated == 2
    assert drawn == ["healthy", "acidosis", "hie"]


def test_the_violin_interior_draws_the_quartiles_and_the_adjacent_values() -> None:
    r"""The mark inside a violin is what a reader takes the middle half off, so it has to be the
    quartiles rather than anything that merely looks like them.

    The sample carries one extreme value, and that is what makes the test non-vacuous: it separates
    the two whisker conventions. An implementation drawing the **range** reaches $100$; Tukey's
    adjacent value stops at the furthest observation inside the fence, which is $9$.
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
            float(line.get_ydata()[0]) for line in axes[0, 0].lines if line.get_marker() == "o"
        ]
    finally:
        shared_figures.plt.close(figure)

    # The whisker first, then the inter-quartile bar: $Q_1 = 3.25$, $Q_3 = 7.75$ on this sample.
    assert spans == [(1.0, 9.0), (3.25, 7.75)]
    assert medians == [5.5]


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


def test_render_figure_writes_the_file_and_leaves_no_open_figure(tmp_path) -> None:
    shared_figures.plt.close("all")
    figure, axes = figures_seam.new_figure(1)
    figures_seam.histogram_panel(axes[0, 0], np.linspace(0.0, 1.0, 32), title="pred_gap")

    path = figures_seam.render_figure(figure, tmp_path / "pred_gap")

    assert Path(path).is_file() and Path(path).stat().st_size > 0
    assert shared_figures.plt.get_fignums() == []


def test_importing_the_seam_does_not_restyle_the_process() -> None:
    """In a subprocess: this session imported the module long ago, so an in-process comparison
    would pass whatever the module does at import time."""
    source = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "before = dict(plt.rcParams)\n"
        "from teb_vae.lag_attn_cfs.eval import figures_seam\n"
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


# =================================================================================================
# The palette
# =================================================================================================
def test_each_clinical_class_carries_its_conventional_colour() -> None:
    """Both the literal and the property behind it. The literal makes a change deliberate; the
    channel test is what a replacement palette still has to satisfy to be the right one."""
    colours = figures_seam.CLINICAL_CLASS_COLORS

    assert colours == {"healthy": "#2E8B57", "acidosis": "#E8A33D", "hie": "#C0392B"}
    red, green, blue = (
        {name: int(value[index:index + 2], 16) for name, value in colours.items()}
        for index in (1, 3, 5)
    )
    assert green["healthy"] > red["healthy"] and green["healthy"] > blue["healthy"]
    assert red["hie"] > green["hie"] and red["hie"] > blue["hie"]
    # Amber is red-plus-green with little blue, which is what separates it from both neighbours.
    assert red["acidosis"] > blue["acidosis"] and green["acidosis"] > blue["acidosis"]


def test_the_palette_is_this_packages_own_rather_than_the_shared_default() -> None:
    """Asserted by value, and the cost is asserted with it: ``utils.style``'s table paints healthy
    blue and is shared with two other projects, so a reader putting an evaluation figure beside a
    training-callback figure of the same cohort must read the legend rather than the hue."""
    from utils.style import CLASS_COLORS_DEFAULT

    assert figures_seam.CLINICAL_CLASS_COLORS != dict(CLASS_COLORS_DEFAULT)
    shared_healthy = dict(CLASS_COLORS_DEFAULT).get("healthy")
    mine = figures_seam.CLINICAL_CLASS_COLORS["healthy"]
    assert shared_healthy is None or shared_healthy != mine


def test_every_canonical_subgroup_is_a_shade_of_its_own_class() -> None:
    """The property the eight-cohort figures are readable because of: a violin's hue says which
    class it belongs to before its label is read."""
    for group in labels.CANONICAL_SUBGROUPS:
        shade = figures_seam.SUBGROUP_COLORS[group]
        red, green, blue = (int(shade[index:index + 2], 16) for index in (1, 3, 5))
        if group.startswith("healthy"):
            assert green > red and green > blue, group
        elif group.startswith("acidosis"):
            assert red > blue and green > blue, group
        else:
            assert red > green and red > blue, group


def test_the_subgroups_of_one_class_are_distinguishable_from_each_other() -> None:
    """A shading range that collapsed would give four identical green violins, which is worse than
    four unrelated hues: the figure would read as one cohort drawn four times."""
    healthy = [
        figures_seam.SUBGROUP_COLORS[name]
        for name in EXPECTED_SUBGROUP_ORDER
        if name.startswith("healthy")
    ]

    assert len(set(healthy)) == len(healthy) == 4
    # Monotone in luminance across the canonical order, so the shading itself carries the order.
    luminance = [sum(int(value[index:index + 2], 16) for index in (1, 3, 5)) for value in healthy]
    assert luminance == sorted(luminance, reverse=True)


def test_a_cohort_keeps_its_colour_whichever_others_a_figure_contains() -> None:
    """Order-independence is what lets two figures of overlapping cohorts be compared. The shared
    palette assigns colours in arrival order for anything it does not know, and that is exactly the
    failure this table replaces for the eleven labels it does."""
    every = list(figures_seam.CLINICAL_CLASS_COLORS) + EXPECTED_SUBGROUP_ORDER
    resolved = figures_seam.group_colors(every)

    assert figures_seam.group_colors(list(reversed(every))) == resolved
    for name in every:
        assert figures_seam.group_colors([name]) == {name: resolved[name]}


def test_an_unknown_cohort_still_receives_a_colour() -> None:
    """A non-canonical shard stem must be drawn, not dropped, so it falls back to the shared
    palette rather than to ``None`` -- which matplotlib would read as "use the default"."""
    resolved = figures_seam.group_colors(["healthy", "not_a_canonical_shard"])

    assert set(resolved) == {"healthy", "not_a_canonical_shard"}
    assert resolved["not_a_canonical_shard"].startswith("#")
