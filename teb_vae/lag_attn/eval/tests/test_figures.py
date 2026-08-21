"""Tests for the extracted figure primitives and the eval plotting panels.

The extraction's real regression guard is ``teb_vae/lag_attn/tests/test_plotting_figure.py`` and
``test_plotting_callback.py``, which exercise the training figures end to end and pass
unmodified or the lift changed behaviour. What is asserted here is the part those cannot see:
that there is genuinely one copy of each helper rather than two, that the new module drags no
framework in, and that the generic panels survive the empty and all-``NaN`` inputs a real run
will hand them.

Every figure test closes its figure in a ``finally``. A leaked figure is not a failure of the
test that leaked it -- it is a memory growth that surfaces somewhere else entirely.
"""
from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from teb_vae.lag_attn import figure_primitives, plotting
from teb_vae.lag_attn.eval import figures

from .conftest import _REPO_ROOT


# ---------------------------------------------------------------------------
# The extraction itself
# ---------------------------------------------------------------------------
LIFTED = (
    "to_numpy",
    "kld_per_dim_np",
    "time_axes",
    "attach_lag_seconds_axis",
    "shade_warmup",
    "average_forecast_per_channel",
    "concat_single_forecasts",
    "stack_feature_blocks",
    "safe_vabs",
    "future_target",
)


@pytest.mark.parametrize("name", LIFTED)
def test_plotting_imports_each_helper_rather_than_defining_it(name: str) -> None:
    """One copy in the tree, not two that a test would have to keep proving identical."""
    assert hasattr(plotting, name), f"plotting.py no longer exposes {name}"
    assert getattr(plotting, name) is getattr(figure_primitives, name), (
        f"plotting.{name} is not the same object as figure_primitives.{name}, so the extraction "
        f"left a second definition behind"
    )


@pytest.mark.parametrize("name", LIFTED)
def test_no_helper_is_redefined_in_plotting(name: str) -> None:
    """An import plus a redefinition would pass the identity check only until someone edits one."""
    source = Path(inspect.getfile(plotting)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert name not in defined


def test_figure_primitives_imports_no_framework() -> None:
    """The whole point of the lift: importable from eval without dragging a callback module in."""
    source = Path(inspect.getfile(figure_primitives)).read_text(encoding="utf-8")
    forbidden = ("lightning", "pytorch_lightning", "matplotlib", "teb_vae.lag_attn.config")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not any(name.startswith(bad) for bad in forbidden), (
                f"figure_primitives imports {name!r}, which defeats the extraction"
            )


def test_the_colour_literals_that_differ_from_utils_style_are_preserved() -> None:
    """Two of the eight genuinely differ, and the figures depend on these hues.

    Sourcing them from ``utils.style`` instead would silently restyle every lag figure.
    """
    from utils import style

    assert figure_primitives.COLOR_PURPLE == "#5642EB" != style.COLOR_PURPLE
    assert figure_primitives.COLOR_BLACK == "#000000" != style.COLOR_BLACK
    # The other six are byte-identical, which is why only these two are worth stating.
    assert figure_primitives.COLOR_BLUE == style.COLOR_BLUE
    assert figure_primitives.COLOR_LIGHT_GRAY == style.COLOR_LIGHT_GRAY


def test_future_target_has_the_shape_the_forecast_is_scored_against() -> None:
    """$(B, T - H_d, H_d, c_y)$, and anchor $t$ holds ``Y[t+1 : t+1+H_d]``."""
    torch.manual_seed(0)
    y_st, y_ph = torch.randn(2, 10, 43), torch.randn(2, 10, 66)
    target = figure_primitives.future_target(y_st, y_ph, horizon=3)
    assert target.shape == (2, 7, 3, 109)
    combined = torch.cat([y_st, y_ph], dim=-1)
    assert torch.equal(target[:, 4, 0], combined[:, 5])
    assert torch.equal(target[:, 4, 2], combined[:, 7])


def test_stack_feature_blocks_returns_the_row_index_of_the_last_top_channel() -> None:
    """``C_top - 1``; a caller drawing the boundary adds $0.5$, which is where they meet."""
    top, bottom = np.zeros((4, 10)), np.zeros((6, 10))
    stacked, separator = figure_primitives.stack_feature_blocks(top, bottom)
    assert stacked.shape == (10, 10)
    assert separator == 3
    assert figure_primitives.stack_feature_blocks(top, None) == (top, None)


def test_safe_vabs_never_returns_zero() -> None:
    """A ``vmax`` of $0$ renders every cell the same colour, which reads as an empty panel."""
    assert figure_primitives.safe_vabs(np.zeros(5)) == 1.0
    assert figure_primitives.safe_vabs(np.full(5, np.nan)) == 1.0
    assert figure_primitives.safe_vabs(np.array([-3.0, 1.0, np.inf])) == 3.0


# ---------------------------------------------------------------------------
# Style discipline
# ---------------------------------------------------------------------------
def test_importing_figures_does_not_mutate_global_rcparams() -> None:
    r"""``apply_publication_style`` is called once at pipeline start, never as an import side effect.

    An import that restyled would silently change any other figure produced in the same process,
    including another test's.

    Measured in a **subprocess**, for two independent reasons. This session imported ``figures``
    long before this test runs, so an in-process check can only observe a *re-execution* of the
    module body, not the real first import. And the way to force that re-execution --
    ``importlib.reload`` -- re-executes into the same module object, replacing every public
    callable with a fresh function object for the rest of the session; the sibling seams bound
    the originals at *their* import time, so
    ``teb_vae/lag_attn_cfs/tests/test_eval_figures.py``'s
    ``assert figures_seam.render_to_pdf is shared_figures.render_to_pdf`` then fails for a reason
    that has nothing to do with the seam. A subprocess measures the true import and leaves this
    process's module graph untouched.
    """
    source = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "before = dict(plt.rcParams)\n"
        "from teb_vae.lag_attn.eval import figures\n"
        "print(','.join(sorted(k for k, v in plt.rcParams.items() if before.get(k) != v)))\n"
        # And the styling is reachable, so it is opt-in rather than simply absent.
        "figures.configure_figure_style()\n"
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
        f"importing figures.py changed rcParams {completed.stdout.strip()}; styling must be an "
        f"explicit call at pipeline start, not an import side effect"
    )


def test_the_behavioural_helpers_come_from_utils_style() -> None:
    """Imported, not reimplemented -- asserted by identity."""
    from utils import style

    assert figures.SAVE_DPI is style.SAVE_DPI
    assert figures.style_axes is style.style_axes
    assert figures.save_figure is style.save_figure


# ---------------------------------------------------------------------------
# Generic panels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "values", [np.array([]), np.full(20, np.nan), np.array([np.nan, np.inf, -np.inf])]
)
def test_histogram_panel_survives_empty_and_all_nan_input(values: np.ndarray) -> None:
    """An analysis that legitimately found nothing must not take down a multi-hour run."""
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.histogram_panel(axes[0, 0], values, title="empty")
        assert drawn == 0
        assert figures.EMPTY_NOTE in [text.get_text() for text in axes[0, 0].texts]
    finally:
        plt.close(fig)


def test_histogram_panel_draws_and_reports_its_finite_count() -> None:
    """The return value is what lets a caller record how much of a capped draw contributed."""
    values = np.concatenate([np.random.default_rng(0).normal(size=50), [np.nan, np.inf]])
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.histogram_panel(axes[0, 0], values, title="t", xlabel="x", reference=0.0)
        assert drawn == 50
        assert axes[0, 0].has_data()
    finally:
        plt.close(fig)


def test_ribbon_plot_survives_an_all_nan_column() -> None:
    """The warm-up anchors are masked to ``NaN`` by construction, in every real profile."""
    curves = np.random.default_rng(1).normal(size=(8, 6))
    curves[:, :2] = np.nan
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.ribbon_plot(axes[0, 0], np.arange(6), curves, title="profile")
        assert drawn == 4
        assert axes[0, 0].has_data()
    finally:
        plt.close(fig)


def test_ribbon_plot_survives_entirely_empty_input() -> None:
    fig, axes = figures.new_figure(1)
    try:
        assert figures.ribbon_plot(axes[0, 0], np.array([]), np.zeros((0, 0))) == 0
    finally:
        plt.close(fig)


def test_heatmap_with_colorbar_draws_and_marks_the_block_separator() -> None:
    fig, axes = figures.new_figure(1)
    try:
        field = np.random.default_rng(2).normal(size=(12, 30))
        image = figures.heatmap_with_colorbar(
            fig, axes[0, 0], field, title="residual", separator_row=4, colorbar_label="v"
        )
        assert image is not None
        assert axes[0, 0].has_data()
        # Symmetric limits about zero, for a signed field.
        assert image.get_clim()[0] == pytest.approx(-image.get_clim()[1])
    finally:
        plt.close(fig)


def test_heatmap_colormap_follows_the_symmetry_of_the_field() -> None:
    """A non-negative field must not be drawn on a diverging colormap.

    ``bwr`` puts white at the midpoint of the range, so a non-negative field renders its
    *smallest* values saturated blue and its mid-range white: the best-forecast channel looks
    extreme and the mediocre one looks neutral, exactly inverting the at-a-glance ranking a
    heatmap exists to give. The colourbar stays correct, so nothing in the numbers gives it away.
    Tying the default to ``symmetric`` is what stops the two from drifting apart at a call site.
    """
    fig, axes = figures.new_figure(2)
    try:
        signed = np.random.default_rng(3).normal(size=(6, 10))
        diverging = figures.heatmap_with_colorbar(fig, axes[0, 0], signed, symmetric=True)
        assert diverging is not None
        assert diverging.get_cmap().name == "bwr"

        non_negative = np.abs(np.random.default_rng(4).normal(size=(6, 10)))
        sequential = figures.heatmap_with_colorbar(fig, axes[1, 0], non_negative, symmetric=False)
        assert sequential is not None
        assert sequential.get_cmap().name != "bwr", (
            "a non-negative field is being drawn on a diverging colormap"
        )
        # An explicit choice still wins over the default.
        explicit = figures.heatmap_with_colorbar(
            fig, axes[1, 0], non_negative, symmetric=False, cmap="viridis"
        )
        assert explicit is not None and explicit.get_cmap().name == "viridis"
    finally:
        plt.close(fig)


def test_heatmap_with_colorbar_survives_an_all_nan_field() -> None:
    fig, axes = figures.new_figure(1)
    try:
        assert figures.heatmap_with_colorbar(fig, axes[0, 0], np.full((4, 4), np.nan)) is None
        assert figures.EMPTY_NOTE in [text.get_text() for text in axes[0, 0].texts]
    finally:
        plt.close(fig)


def test_heatmap_gives_a_constant_field_a_usable_colour_range() -> None:
    """A degenerate range renders flat, which is indistinguishable from an empty panel."""
    fig, axes = figures.new_figure(1)
    try:
        image = figures.heatmap_with_colorbar(
            fig, axes[0, 0], np.full((4, 4), 3.0), symmetric=False
        )
        assert image is not None
        low, high = image.get_clim()
        assert high > low
    finally:
        plt.close(fig)


def test_as_columns_refuses_a_name_count_that_does_not_match() -> None:
    """Silently zipping the shorter of the two would drop columns off every emitted CSV."""
    with pytest.raises(ValueError, match="column name"):
        figures.as_columns(np.zeros((4, 3)), ["a", "b"])


def test_render_to_pdf_writes_a_pdf_and_closes_the_figure(tmp_path) -> None:
    """PDF only, matching the repository convention and the existing figure tests."""
    fig, axes = figures.new_figure(1)
    axes[0, 0].plot([0, 1], [0, 1])
    path = tmp_path / "panel.pdf"
    figures.render_to_pdf(fig, path)
    assert path.is_file() and path.stat().st_size > 0
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# The binned violin panel
# ---------------------------------------------------------------------------
def _body_centres(ax) -> list:
    """The x coordinate each violin body is centred on, in ascending order.

    A violin body is the only ``PolyCollection`` these panels draw -- the interior marks are line
    collections and the significance bars are patches -- and a kernel density is symmetric about
    its position, so the midpoint of a body's vertex span **is** the position it was drawn at.
    """
    from matplotlib.collections import PolyCollection

    centres = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            for path in collection.get_paths():
                x = np.asarray(path.vertices, dtype=np.float64)[:, 0]
                centres.append((float(x.min()) + float(x.max())) / 2.0)
    return sorted(centres)


def _plotted_points(ax) -> list:
    """The ``(x, y)`` pairs drawn as bare markers, which is how a too-thin cell is shown.

    Discriminated from the interior's median dot by its linestyle: the dot is drawn with the
    default solid style and these with ``'none'``. The legend proxies share that style and carry
    no data, so they are excluded by the emptiness check rather than by name.
    """
    points = []
    for line in ax.lines:
        if line.get_linestyle() == "None" and len(line.get_xdata()):
            points.extend(zip(list(line.get_xdata()), list(line.get_ydata())))
    return points


def _samples(*, groups, per_window, spread: float = 1.0) -> list:
    """One ``{group: values}`` mapping per window, each group's cell holding ``per_window`` values.

    Args:
        groups: Group labels.
        per_window: Values per cell, per window. ``0`` leaves that cell out entirely.
        spread: Multiplies the within-cell range, so a caller can make a cell constant.

    Returns:
        The sequence :func:`figures.binned_violin_panel` takes.
    """
    windows = []
    for count in per_window:
        cell = {}
        for offset, group in enumerate(groups):
            if count:
                cell[group] = np.arange(float(count)) * spread + float(offset) * 10.0
        windows.append(cell)
    return windows


def test_the_dodge_places_each_group_symmetrically_inside_its_window() -> None:
    r"""$x = c + (i - \frac{k-1}{2})\frac{w}{k+1}$, for $k = 2$ and $k = 3$.

    Asserted numerically rather than eyeballed: a dodge that drifted would put a cohort's body
    over the neighbouring window's, and every value on the page would still be correct.
    """
    for groups, expected in (
        (["a", "b"], [-1.0 / 6.0, 1.0 / 6.0]),
        (["a", "b", "c"], [-0.25, 0.0, 0.25]),
    ):
        fig, axes = figures.new_figure(1)
        try:
            drawn = figures.binned_violin_panel(
                axes[0, 0], _samples(groups=groups, per_window=[5]), [0.0],
                groups=groups, bin_width=1.0, min_body_size=3,
            )
            assert drawn == len(groups)
            assert _body_centres(axes[0, 0]) == pytest.approx(expected)
        finally:
            plt.close(fig)


def test_a_group_absent_from_one_window_leaves_a_gap_rather_than_shifting_its_neighbours() -> None:
    """The dodge is computed from the figure's group count, not the window's. Recentring the
    survivors would make a cohort's position mean something different in different windows, and
    the trajectory could no longer be read across them."""
    groups = ["a", "b", "c"]
    windows = _samples(groups=groups, per_window=[5])
    del windows[0]["b"]

    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.binned_violin_panel(
            axes[0, 0], windows, [0.0], groups=groups, bin_width=1.0, min_body_size=3
        )
        assert drawn == 2
        # 'a' and 'c' keep the outer slots the three-group case gave them.
        assert _body_centres(axes[0, 0]) == pytest.approx([-0.25, 0.25])
    finally:
        plt.close(fig)


def test_a_cell_too_thin_for_a_density_is_drawn_as_its_own_values() -> None:
    """matplotlib evaluates the kernel between the data's own extremes, so a body over two values
    is a shape the smoother invented -- and the caller's test excludes that cell at the same
    threshold, so the figure and the test agree about which cells carry evidence."""
    fig, axes = figures.new_figure(1)
    try:
        figures.binned_violin_panel(
            axes[0, 0], _samples(groups=["a"], per_window=[2]), [0.0],
            groups=["a"], bin_width=1.0, min_body_size=3,
        )
        assert _body_centres(axes[0, 0]) == []
        assert len(_plotted_points(axes[0, 0])) == 2
    finally:
        plt.close(fig)

    fig, axes = figures.new_figure(1)
    try:
        figures.binned_violin_panel(
            axes[0, 0], _samples(groups=["a"], per_window=[3]), [0.0],
            groups=["a"], bin_width=1.0, min_body_size=3,
        )
        assert len(_body_centres(axes[0, 0])) == 1
        assert _plotted_points(axes[0, 0]) == []
    finally:
        plt.close(fig)


def test_a_constant_cell_draws_its_points_rather_than_raising() -> None:
    """``gaussian_kde`` inverts a covariance that is singular when every value is equal, and
    raises. Three recordings that happened to score identically would otherwise take the figure
    down at the run's final step."""
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.binned_violin_panel(
            axes[0, 0], [{"a": np.full(5, 2.5)}], [0.0],
            groups=["a"], bin_width=1.0, min_body_size=3,
        )
        assert drawn == 1
        assert _body_centres(axes[0, 0]) == []
        assert len(_plotted_points(axes[0, 0])) == 5
    finally:
        plt.close(fig)


def test_every_cell_is_annotated_with_the_count_behind_it() -> None:
    """A body can move because the population changed rather than because the quantity did, and
    the count is the only thing that says which."""
    fig, axes = figures.new_figure(1)
    try:
        figures.binned_violin_panel(
            axes[0, 0], [{"a": np.arange(5.0), "b": np.arange(3.0)}], [0.0],
            groups=["a", "b"], bin_width=1.0, min_body_size=3,
        )
        assert sorted(text.get_text() for text in axes[0, 0].texts) == ["3", "5"]
    finally:
        plt.close(fig)


def test_the_bodies_take_the_callers_palette() -> None:
    """A cohort keeps one colour across every figure of a run, and the palette that decides it is
    the caller's package's rather than this module's default."""
    from matplotlib.colors import to_rgba
    from matplotlib.collections import PolyCollection

    palette = {"a": "#2E8B57", "b": "#C0392B"}
    fig, axes = figures.new_figure(1)
    try:
        figures.binned_violin_panel(
            axes[0, 0], _samples(groups=["a", "b"], per_window=[5]), [0.0],
            groups=["a", "b"], bin_width=1.0, min_body_size=3, colors=palette,
        )
        faces = [
            tuple(collection.get_facecolor()[0][:3])
            for collection in axes[0, 0].collections
            if isinstance(collection, PolyCollection)
        ]
        assert faces == [to_rgba(palette["a"])[:3], to_rgba(palette["b"])[:3]]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "windows, groups",
    [([], []), ([{}, {}], ["a"]), ([{"a": np.full(4, np.nan)}], ["a"])],
)
def test_the_binned_panel_survives_empty_and_all_nan_input(windows, groups) -> None:
    """An analysis that legitimately found nothing must not take down a multi-hour run."""
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.binned_violin_panel(
            axes[0, 0], windows, [0.0] * len(windows),
            groups=groups, bin_width=0.5, min_body_size=3,
        )
        assert drawn == 0
        assert figures.EMPTY_NOTE in [text.get_text() for text in axes[0, 0].texts]
    finally:
        plt.close(fig)


def test_the_binned_panel_refuses_misaligned_windows_and_centres() -> None:
    """Zipping the shorter of the two would draw one window's values at another's coordinate,
    with every axis label still saying they were where they belong."""
    fig, axes = figures.new_figure(1)
    try:
        with pytest.raises(ValueError, match="positionally aligned"):
            figures.binned_violin_panel(
                axes[0, 0], _samples(groups=["a"], per_window=[5, 5]), [0.0],
                groups=["a"], bin_width=1.0, min_body_size=3,
            )
    finally:
        plt.close(fig)


def test_the_binned_panel_counts_only_the_cells_that_had_data() -> None:
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.binned_violin_panel(
            axes[0, 0], _samples(groups=["a", "b"], per_window=[5, 0, 4]), [0.0, 0.5, 1.0],
            groups=["a", "b"], bin_width=0.5, min_body_size=3,
        )
        assert drawn == 4
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# The significance strip
# ---------------------------------------------------------------------------
def _bars(ax) -> list:
    """``(x centre, height, width)`` per drawn bar."""
    return [
        (patch.get_x() + patch.get_width() / 2.0, patch.get_height(), patch.get_width())
        for patch in ax.patches
    ]


def test_the_strip_draws_one_bar_per_testable_window_against_the_threshold() -> None:
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.significance_strip(
            axes[0, 0], [0.25, 0.75, 1.25], [0.001, 0.5, 0.02], alpha=0.05, bin_width=0.5
        )
        assert drawn == 3
        centres, heights, widths = zip(*_bars(axes[0, 0]))
        assert list(centres) == pytest.approx([0.25, 0.75, 1.25])
        assert list(heights) == pytest.approx([3.0, -np.log10(0.5), -np.log10(0.02)])
        # A fraction of a window, so the bar sits inside the window it describes.
        assert list(widths) == pytest.approx([0.5 * figures.SIGNIFICANCE_BAR_FRACTION] * 3)

        threshold = [
            line for line in axes[0, 0].lines
            if len(np.atleast_1d(line.get_ydata())) == 2
            and np.atleast_1d(line.get_ydata())[0] == np.atleast_1d(line.get_ydata())[1]
        ]
        assert threshold, "no threshold line was drawn"
        assert float(np.atleast_1d(threshold[0].get_ydata())[0]) == pytest.approx(-np.log10(0.05))
        assert "alpha = 0.05" in threshold[0].get_label()
        assert "Holm" in threshold[0].get_label()
    finally:
        plt.close(fig)


def test_a_p_value_of_exactly_zero_stays_on_the_page() -> None:
    """A rank test returns an exact zero at large $n$, and an infinite bar would rescale the axis
    until every other window read as a flat line -- so the strongest evidence would erase the
    evidence everywhere else."""
    fig, axes = figures.new_figure(1)
    try:
        figures.significance_strip(
            axes[0, 0], [0.0, 1.0], [0.0, 0.04], alpha=0.05, bin_width=1.0
        )
        heights = [height for _, height, _ in _bars(axes[0, 0])]
        assert np.isfinite(heights).all()
        assert heights[0] == pytest.approx(-np.log10(figures.P_VALUE_FLOOR))
        assert np.isfinite(axes[0, 0].get_ylim()).all()
    finally:
        plt.close(fig)


def test_an_untestable_window_gets_a_mark_of_its_own_rather_than_a_bar() -> None:
    """Zero height is "no evidence"; a window that could not be tested is a different statement.
    It gets no bar -- and a mark at zero, because at any realistic window count an absent bar and
    an absent window are otherwise the same empty stretch of axis."""
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.significance_strip(
            axes[0, 0], [0.0, 1.0, 2.0], [0.01, float("nan"), 0.2], alpha=0.05, bin_width=1.0
        )
        assert drawn == 2
        assert [centre for centre, _, _ in _bars(axes[0, 0])] == pytest.approx([0.0, 2.0])

        marks = [line for line in axes[0, 0].lines if line.get_marker() == "x"]
        assert len(marks) == 1
        assert list(np.atleast_1d(marks[0].get_xdata())) == pytest.approx([1.0])
        assert list(np.atleast_1d(marks[0].get_ydata())) == pytest.approx([0.0])
        assert marks[0].get_label() == "not testable"
    finally:
        plt.close(fig)


def test_a_window_tested_at_p_one_is_distinguishable_from_an_untested_one() -> None:
    """The pair the mark above exists to separate: both draw nothing visible at this scale, and
    only the mark says which is which."""
    fig, axes = figures.new_figure(1)
    try:
        figures.significance_strip(
            axes[0, 0], [0.0, 1.0], [1.0, float("nan")], alpha=0.05, bin_width=1.0
        )
        bars = _bars(axes[0, 0])
        marks = [line for line in axes[0, 0].lines if line.get_marker() == "x"]

        # The tested window has a bar of zero height; the untested one has a mark and no bar.
        assert [centre for centre, _, _ in bars] == pytest.approx([0.0])
        assert [height for _, height, _ in bars] == pytest.approx([0.0])
        assert list(np.atleast_1d(marks[0].get_xdata())) == pytest.approx([1.0])
    finally:
        plt.close(fig)


def test_a_strip_with_nothing_testable_says_so_and_still_shows_where_the_windows_were() -> None:
    fig, axes = figures.new_figure(1)
    try:
        drawn = figures.significance_strip(
            axes[0, 0], [0.0, 1.0], [float("nan")] * 2, alpha=0.05, bin_width=1.0
        )
        assert drawn == 0
        assert figures.EMPTY_NOTE in [text.get_text() for text in axes[0, 0].texts]
        assert _bars(axes[0, 0]) == []
        marks = [line for line in axes[0, 0].lines if line.get_marker() == "x"]
        assert list(np.atleast_1d(marks[0].get_xdata())) == pytest.approx([0.0, 1.0])
    finally:
        plt.close(fig)


def test_the_strip_refuses_misaligned_centres_and_p_values() -> None:
    fig, axes = figures.new_figure(1)
    try:
        with pytest.raises(ValueError, match="positionally aligned"):
            figures.significance_strip(
                axes[0, 0], [0.0, 1.0], [0.01], alpha=0.05, bin_width=1.0
            )
    finally:
        plt.close(fig)
