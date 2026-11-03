r"""Check rendered cell centres against the stored sample timestamps.

These assertions read the actual Matplotlib artists. They catch a half-step
translation even when the arrays, row ordering, and axis limits are unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from teb_vae.lag_attn.figure_primitives import sample_cell_edges
from teb_vae.lag_attn_cfs.eval.analyses import attention as cfs_attention
from teb_vae.lag_attn_cfs.sample_page import COMPACT_PAGE_ROWS
from teb_vae.lag_attn_cfs.tests.test_sample_page import _render
from teb_vae.lag_attn_rws import sample_page as shared_page
from teb_vae.lag_attn_rws.eval.analyses import attention as rws_attention
from teb_vae.lag_attn_transformer_rws.eval.analyses import encoder_attention


def _centres(left: float, right: float, count: int) -> np.ndarray:
    r"""Recover the $N$ cell centres from an image's two outside edges.

    Args:
        left: First outside edge, in plotted coordinates.
        right: Last outside edge, in plotted coordinates.
        count: Number $N$ of cells on this axis.

    Returns:
        Coordinates at which Matplotlib draws the cell centres.
    """
    return left + (np.arange(count) + 0.5) * (right - left) / count


@pytest.mark.parametrize("count,spacing,first", [(1, 4.0, 0.0), (300, 4.0, 0.0),
                                               (156, 4.0, 536.0), (3, 0.25, -1.0)])
def test_sample_edges_place_cells_at_the_samples(count, spacing, first):
    """Cover one sample, production length, a cropped prefix, and another unit."""
    edges = sample_cell_edges(count, spacing, first=first)
    np.testing.assert_allclose(_centres(*edges, count), first + np.arange(count) * spacing)


@pytest.mark.parametrize("compact", [False, True])
def test_sample_page_heatmaps_share_the_line_timestamps(task, stub_batch, compact):
    """Check the full callback page and the compact evaluation page on the same grid."""
    module = task()
    figure = _render(module, stub_batch, **({"rows": COMPACT_PAGE_ROWS} if compact else {}))
    try:
        geometry = module.orig_model.geometry
        spacing = geometry.raw_len / 4.0 / geometry.t
        checked = 0
        for axis in figure.axes:
            # The untitled forecast-error inset has a horizon-index axis, not segment time.
            if not axis.get_title() or not axis.images:
                continue
            image = axis.images[0]
            count = image.get_array().shape[1]
            start = 0 if count == geometry.t else geometry.warmup
            np.testing.assert_allclose(
                _centres(*image.get_extent()[:2], count),
                np.arange(start, start + count) * spacing,
                err_msg=axis.get_title(),
            )
            assert axis.get_xlim() == pytest.approx((0.0, geometry.t * spacing))
            checked += 1
        assert checked >= 3
    finally:
        plt.close(figure)


@pytest.mark.parametrize("compact", [False, True])
def test_sample_page_shades_abut_the_trained_cells(task, stub_batch, compact):
    r"""Check that the grey margins end and start on cell edges, not on step instants.

    A trained row's image spans $[(w - \tfrac12)\Delta, (T_{\mathrm{valid}} - \tfrac12)\Delta]$.
    The warm-up shade must end at its left edge and the tail shade must start at its right
    edge; a shade drawn to $w\Delta$ covers half of the first trained column.
    """
    module = task()
    figure = _render(module, stub_batch, **({"rows": COMPACT_PAGE_ROWS} if compact else {}))
    try:
        geometry = module.orig_model.geometry
        spacing = geometry.raw_len / 4.0 / geometry.t
        t_max = geometry.t * spacing
        warmup_edge = (geometry.warmup - 0.5) * spacing
        checked = 0
        for axis in figure.axes:
            if not axis.get_title() or not axis.images:
                continue
            count = axis.images[0].get_array().shape[1]
            if count == geometry.t:
                continue
            left, right = axis.images[0].get_extent()[:2]
            spans = [
                (float(patch.get_x()), float(patch.get_x() + patch.get_width()))
                for patch in axis.patches
            ]
            assert (0.0, pytest.approx(warmup_edge)) in spans, axis.get_title()
            assert left == pytest.approx(warmup_edge), axis.get_title()
            if right < t_max - 1e-9:
                assert (pytest.approx(right), pytest.approx(t_max)) in spans, axis.get_title()
            checked += 1
        assert checked >= 3
    finally:
        plt.close(figure)


def test_fitted_limits_follow_the_sample_inside_the_physiological_range():
    """Fit the readings, skip the gap sentinel, keep a minimum span and respect the bounds."""
    # A 120-160 bpm trace with dropouts stored as 0 bpm: the zeros are not readings.
    trace = np.concatenate([np.linspace(120.0, 160.0, 50), np.zeros(5), [np.nan]])
    low, high = shared_page.fitted_limits(
        trace, min_span=shared_page.MIN_TRACE_SPAN["fhr"],
        bounds=shared_page.RAW_TRACE_LIMITS["fhr"],
    )
    assert (low, high) == pytest.approx((118.0, 162.0))

    # A flat trace is widened about its centre to the span floor, not stretched to fill the row.
    assert shared_page.fitted_limits(
        np.full(10, 140.0), min_span=30.0, bounds=(50.0, 210.0)
    ) == pytest.approx((125.0, 155.0))

    # Near a bound, the range is moved inside it rather than cut.
    assert shared_page.fitted_limits(
        np.full(10, 2.0), min_span=20.0, bounds=(0.0, 100.0)
    ) == pytest.approx((0.0, 20.0))

    # A non-negative quantity keeps its floor and pads only the top.
    assert shared_page.fitted_limits(np.array([0.2, 1.0]), floor=0.0) == pytest.approx((0.0, 1.05))
    assert shared_page.fitted_limits(np.full(3, np.nan)) is None


def test_sample_page_kld_row_is_fitted_to_the_sample(task, stub_batch):
    """A run-wide limit passed for the $K_t$ row no longer sets its y-axis; the sample does."""
    module = task()
    figure = _render(module, stub_batch, row_limits={"kld_total": 1.0e6})
    try:
        axis = next(ax for ax in figure.axes if ax.get_title().startswith("$K_t$"))
        drawn = np.asarray(axis.lines[0].get_ydata(), dtype=float)
        assert axis.get_ylim() == pytest.approx(shared_page.fitted_limits(drawn, floor=0.0))
        assert axis.get_ylim()[1] < 1.0e6
    finally:
        plt.close(figure)


@pytest.mark.parametrize("analysis", [cfs_attention, rws_attention])
@pytest.mark.parametrize("steps,lags", [(12, 9), (1, 1)])
def test_eval_attention_cells_locate_the_actual_anchor_and_lag(analysis, steps, lags):
    """Locate a planted last-anchor/lag-zero peak, including a single-cell map."""
    retained = np.zeros((1, steps, 1, lags))
    retained[0, -1, 0, 0] = 1.0
    figure = analysis.build_heatmap_figure(
        retained, row=0, delay_steps=3, geometry={"anchor_floor": 0, "warmup": 0}
    )
    try:
        image = figure.axes[0].images[0]
        left, right, bottom, top = image.get_extent()
        x = _centres(left, right, steps)
        # The shared renderer uses upper origin and reverses the lag rows.
        y = _centres(top, bottom, lags)
        row, column = np.unravel_index(np.argmax(image.get_array()), (lags, steps))
        assert x[column] == pytest.approx((steps - 1) * 4.0)
        assert y[row] == pytest.approx(3 * 4.0)
        np.testing.assert_allclose(x, np.arange(steps) * 4.0)
        np.testing.assert_allclose(y[::-1], (np.arange(lags) + 3) * 4.0)
    finally:
        plt.close(figure)


def test_encoder_attention_centres_match_query_and_key_timestamps():
    """A planted attention entry must sit at its query and key sample coordinates."""
    weights = np.zeros((12, 12))
    weights[7, 2] = 1.0
    result = SimpleNamespace(seq_len=12, heatmaps={("target", 0): weights})
    figure = encoder_attention.build_heatmap_figure(result)
    try:
        image = figure.axes[0].images[0]
        left, right, bottom, top = image.get_extent()
        row, column = np.unravel_index(np.argmax(image.get_array()), weights.shape)
        assert _centres(left, right, 12)[column] == pytest.approx(8.0)
        assert _centres(top, bottom, 12)[row] == pytest.approx(28.0)
    finally:
        plt.close(figure)
