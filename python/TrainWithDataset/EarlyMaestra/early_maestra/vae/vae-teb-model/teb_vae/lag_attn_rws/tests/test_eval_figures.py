r"""The figure-layer properties this package owns rather than inherits from the shared panels.

**The violin interior is the quartiles and Tukey's adjacent values.** Checked here because the
owner of the shared panel does not pin the whisker convention, and a range whisker reads exactly
like an adjacent-value one until an outlier is present.

**Importing must not restyle anything.** ``apply_publication_style`` mutates global ``rcParams``,
so an import-time call silently restyles every other figure produced in the same process --
including one a test is asserting on, and including the training callback's if an evaluation is
ever run in-process beside it. Checked in a subprocess, because this session has imported the
module long before the test runs and an in-process check would pass no matter what.

**An empty panel is a result.** The subgroup heatmap with no Holm survivor says so rather than
arriving as a blank figure that reads as a plotting failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from matplotlib.collections import LineCollection

from teb_vae.lag_attn.eval import figures as shared_figures
from teb_vae.lag_attn_rws.eval import figures_seam

_REPO_ROOT = Path(__file__).resolve().parents[3]


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
