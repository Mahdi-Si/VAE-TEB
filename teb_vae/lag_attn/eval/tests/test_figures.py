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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from teb_vae.lag_attn import figure_primitives, plotting
from teb_vae.lag_attn.eval import figures


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
    """``apply_publication_style`` is called once at pipeline start, never as an import side effect.

    An import that restyled would silently change any other figure produced in the same process,
    including another test's.
    """
    import importlib

    before = dict(plt.rcParams)
    importlib.reload(figures)
    after = dict(plt.rcParams)
    changed = {key for key in before if before[key] != after.get(key)}
    assert not changed, f"importing figures.py changed rcParams: {sorted(changed)}"


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
