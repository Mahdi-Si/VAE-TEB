"""The per-sample diagnostic page: alignment, row set, and non-vacuous content.

The alignment assertions are the point of the two-column gridspec, and they are the ones a
casual refactor breaks: attaching a colorbar to an axes rather than to its reserved slot steals
width from that axes alone, which reads as a cosmetic wobble and is in fact the shared time
axis silently ceasing to line up.

The sabotage case is what makes the rest non-vacuous. Structural assertions on row counts and
titles pass just as happily on a figure drawn from zeros, so one channel's forecast is driven
far out of range and the residual row is required to show it.

Every test closes its figure in a ``finally``: a leaked figure is not a failure of the test that
leaked it, it is memory growth that surfaces somewhere else entirely.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import torch

from teb_vae.lag_attn.eval import sample_figure
from teb_vae.lag_attn.eval.figures import plt

#: Small enough to build in milliseconds, large enough that every row has real structure.
T = 40
H_D = 5
WARMUP = 4
D_Z = 6
N_HEADS = 2
N_LAGS = 7
N_SCATTERING = 3
N_PHASE = 4


def _outputs(seed: int = 0) -> Dict[str, torch.Tensor]:
    """One sample's forward tensors, shaped as the model returns them without a batch axis."""
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator)

    # Rows sum to 1, as an eval()-mode attention pass produces; the te_lag_map identity is
    # built on that and a uniform stand-in would hide a normalisation bug.
    alpha = torch.softmax(normal(T, N_HEADS, N_LAGS), dim=-1)
    kld_per_t = normal(T).abs()
    return {
        "mu_full": normal(T, H_D, N_SCATTERING + N_PHASE),
        "z": normal(T, D_Z),
        "mu_prior": normal(T, D_Z),
        "logvar_prior": torch.zeros(T, D_Z),
        "mu_post": normal(T, D_Z),
        "logvar_post": torch.zeros(T, D_Z),
        "kld_per_t": kld_per_t,
        "attn_weights": alpha,
        # The attribution the model exposes: the per-step KL spread over lags by mean attention.
        "te_lag_map": kld_per_t[:, None] * alpha.mean(dim=1),
    }


def _targets(seed: int = 1):
    """The target feature blocks for one sample."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(T, N_SCATTERING, generator=generator),
        torch.randn(T, N_PHASE, generator=generator),
    )


def _main_axes(figure) -> list:
    """The row axes of a built figure, in page order.

    Selected by carrying a title, which is what ``RowGrid.finalise`` gives every row and gives
    nothing else: the reserved colorbar slots have none, and neither do the ``twinx`` axes the
    raw and $K_t$ rows add -- and a twin sits at its parent's position, so selecting on width
    would silently double-count exactly two rows.
    """
    return [axes for axes in figure.axes if axes.get_title()]


def _build(**overrides: Any):
    """Build a figure at the test geometry, with the raw context present unless overridden."""
    y_st, y_ph = _targets()
    kwargs: Dict[str, Any] = {
        "outputs": _outputs(),
        "y_st": y_st,
        "y_ph": y_ph,
        "fhr_raw": torch.randn(T * sample_figure.DECIMATION),
        "up_raw": torch.randn(T * sample_figure.DECIMATION),
        "warmup": WARMUP,
        "horizon": H_D,
        "guid": "abc123",
        "epoch": 7,
    }
    kwargs.update(overrides)
    return sample_figure.build_sample_figure(**kwargs)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def test_every_main_axes_has_the_same_width_and_shares_the_time_limits():
    """The reserved colorbar column exists for exactly this; without it the widths diverge."""
    figure = _build()
    try:
        boxes = [axes.get_position() for axes in _main_axes(figure)]
        assert len(boxes) == len(sample_figure.ROW_SPECS)

        widths = {round(box.width, 6) for box in boxes}
        assert len(widths) == 1, f"main axes widths diverged: {sorted(widths)}"

        # And the reserved slots really are the narrow column, so the equality above is the
        # gridspec doing its job rather than every axes happening to span the page.
        reserved = [
            axes.get_position().width
            for axes in figure.axes
            if not axes.get_title() and axes.get_position().width < boxes[0].width
        ]
        assert reserved, "no narrow colorbar slot was reserved"
    finally:
        plt.close(figure)


def test_every_row_shares_one_physical_time_axis():
    figure = _build()
    try:
        expected = (0.0, float(T * sample_figure.DECIMATION) / sample_figure.FS_RAW)
        limits = {
            tuple(round(value, 6) for value in axes.get_xlim())
            for axes in _main_axes(figure)
        }
        assert limits == {tuple(round(value, 6) for value in expected)}
    finally:
        plt.close(figure)


def test_the_figure_builds_from_a_stub_row_set():
    """The grid is driven by whatever row list it is handed, not by a hardcoded count."""
    stub = [("a", "A", 1.0), ("b", "B", 2.0)]
    grid = sample_figure.RowGrid(stub, t_max=10.0)
    try:
        for name, _, _ in stub:
            axes, cax = grid.axes(name)
            axes.plot([0.0, 10.0], [0.0, 1.0])
            grid.hide_colorbar(cax)
        assert len(grid.main_axes()) == 2
        assert grid.has("a") and not grid.has("z")
    finally:
        plt.close(grid.figure)


def test_asking_for_a_row_that_was_dropped_raises():
    """Silently drawing nothing would leave the caller believing the row is there."""
    grid = sample_figure.RowGrid(sample_figure.resolve_rows(()), t_max=1.0)
    try:
        with pytest.raises(KeyError):
            grid.axes("raw")
    finally:
        plt.close(grid.figure)


# ---------------------------------------------------------------------------
# Row set and titles
# ---------------------------------------------------------------------------
def test_rows_appear_in_the_declared_order_with_their_title_prefixes():
    """Derived from ROW_SPECS rather than from a second literal list, which would drift."""
    figure = _build()
    try:
        titles = [axes.get_title() for axes in _main_axes(figure)]
        prefixes = [prefix for _, prefix, _ in sample_figure.ROW_SPECS]
        assert len(titles) == len(prefixes)
        for title, prefix in zip(titles, prefixes):
            assert title.startswith(prefix), f"expected {prefix!r} to lead {title!r}"
    finally:
        plt.close(figure)


def test_channel_labels_come_from_the_tensor_shapes_not_from_literals():
    """A hardcoded 109 would keep passing after the dataset's channel selection moved again."""
    figure = _build()
    try:
        titles = " ".join(axes.get_title() for axes in figure.axes)
        assert f"{N_SCATTERING + N_PHASE} channels" in titles
        assert f"$d_z$={D_Z}" in titles
        assert f"$L$={N_LAGS}" in titles
        assert f"$H_d$={H_D}" in titles
    finally:
        plt.close(figure)


def test_the_te_lag_row_states_whether_it_is_an_attribution_or_a_diagnostic():
    """Without head_structured_latent the same picture means something weaker, and only the
    caption can say so."""
    figure = _build(te_lag_label="diagnostic")
    try:
        titles = " ".join(axes.get_title() for axes in figure.axes)
        assert "diagnostic" in titles
    finally:
        plt.close(figure)


def test_forecast_and_target_rows_share_one_colour_range():
    """Two independently scaled heatmaps of the same quantity read as more alike than they are."""
    figure = _build()
    try:
        images = [image for axes in figure.axes for image in axes.get_images()]
        # The first two heatmap rows after the raw line-plot row are forecast and target.
        forecast, target = images[0], images[1]
        assert forecast.get_clim() == target.get_clim()
        # ... and the residual row, which is a different quantity, must not be forced onto it.
        assert images[2].get_clim() != forecast.get_clim()
    finally:
        plt.close(figure)


def test_every_main_axes_actually_drew_something():
    figure = _build()
    try:
        for axes in _main_axes(figure):
            assert axes.has_data(), f"empty panel: {axes.get_title()!r}"
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# Optional row degradation
# ---------------------------------------------------------------------------
def test_a_batch_without_raw_fields_loses_exactly_one_row():
    figure = _build(fhr_raw=None, up_raw=None)
    try:
        main = _main_axes(figure)
        assert len(main) == len(sample_figure.ROW_SPECS) - 1
        titles = [axes.get_title() for axes in main]
        assert not any(title.startswith("Raw FHR") for title in titles)
        # The remaining rows keep their order rather than merely losing one from the middle.
        expected = [prefix for name, prefix, _ in sample_figure.ROW_SPECS if name != "raw"]
        for title, prefix in zip(titles, expected):
            assert title.startswith(prefix)
    finally:
        plt.close(figure)


def test_no_empty_axes_is_left_behind_when_the_raw_row_is_dropped():
    """Dropping the row from the gridspec, not merely leaving its panel blank."""
    figure = _build(fhr_raw=None, up_raw=None)
    try:
        for axes in figure.axes:
            assert axes.has_data() or not axes.get_visible()
    finally:
        plt.close(figure)


def test_resolve_rows_drops_a_row_when_only_one_of_its_inputs_is_present():
    """``fhr`` without ``up`` cannot draw the twin-axis context row."""
    assert [name for name, _, _ in sample_figure.resolve_rows(("fhr",))][0] != "raw"
    assert [name for name, _, _ in sample_figure.resolve_rows(("fhr", "up"))][0] == "raw"


def test_one_raw_field_alone_still_drops_the_row_in_the_built_figure():
    figure = _build(up_raw=None)
    try:
        assert len(_main_axes(figure)) == len(sample_figure.ROW_SPECS) - 1
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# Non-vacuity
# ---------------------------------------------------------------------------
def test_a_sabotaged_channel_shows_up_in_the_residual_row():
    """Structural assertions pass on a figure of zeros; this one does not."""
    outputs = _outputs()
    sabotaged = int(N_SCATTERING + 1)
    outputs["mu_full"][:, :, sabotaged] += 50.0

    figure = _build(outputs=outputs)
    try:
        images = [image for axes in figure.axes for image in axes.get_images()]
        residual = images[2].get_array()
        finite = np.asarray(residual, dtype=np.float64)
        row_magnitude = np.nanmax(np.abs(finite), axis=1)

        assert int(np.nanargmax(row_magnitude)) == sabotaged, (
            "the residual row does not report the channel whose forecast was sabotaged"
        )
        assert row_magnitude[sabotaged] > 10.0
    finally:
        plt.close(figure)


def test_the_lag_seconds_axis_agrees_with_the_pipelines_own_conversion():
    r"""Two figures in one run must not label the same lag differently.

    ``attach_lag_seconds_axis`` maps $\ell \mapsto s\ell + o$ and the pipeline's convention --
    ``metrics.lag_to_seconds`` -- is $s\ell$ on the stored timeline, so both call sites must hand
    the helper an offset of exactly $0$: the dataset builder's UP shift is part of the stored signal
    and is never applied to an axis. What this test rules out is one call site carrying an offset
    and the other not, which would label the same lag two ways inside one page.
    """
    from teb_vae.lag_attn.eval import metrics

    # matplotlib keeps a SecondaryAxis as a child rather than in ``figure.axes``, so the offset
    # is captured at the call instead of read back off the drawn figure.
    captured = []
    real = sample_figure.attach_lag_seconds_axis

    def recording(ax, step_seconds, offset):
        captured.append((float(step_seconds), float(offset)))
        return real(ax, step_seconds, offset)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sample_figure, "attach_lag_seconds_axis", recording)
        figure = _build(step_seconds=4.0)
        plt.close(figure)

    assert len(captured) == 2, "both lag rows must carry a seconds axis"
    for step_seconds, offset in captured:
        assert offset == 0.0
        # attach_lag_seconds_axis maps l -> s*l + offset, so the offset it is handed must make
        # that identical to the pipeline's own conversion.
        for lag in (0.0, 5.0, float(N_LAGS - 1)):
            assert step_seconds * lag + offset == pytest.approx(
                metrics.lag_to_seconds(lag, step_seconds=4.0)
            )


def test_the_page_title_carries_the_guid_and_the_epoch():
    figure = _build()
    try:
        assert "abc123" in figure._suptitle.get_text()
        assert "epoch 7" in figure._suptitle.get_text()
    finally:
        plt.close(figure)
