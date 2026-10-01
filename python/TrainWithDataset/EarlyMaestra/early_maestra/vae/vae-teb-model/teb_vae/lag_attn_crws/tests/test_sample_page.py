r"""The forecast row this model replaces, drawn through the callback that runs it.

The shipped raw rows tile through ``concat_single_forecasts``, which reads its per-anchor block at an
**anchor** index, while this model's forecast is $(A_{\max}, H, R)$ indexed by position in the decoded
set. At the shipped floor the first read is out of range and the callback's broad handler turns a
whole run's diagnostics into one log line; at a smaller floor it draws a real forecast at the wrong
time with no exception anywhere. The input rows and the run-level budget figure are the
causal-feature cell's, bound by the task, and are tested there.

So the page is tested through the **callback** (every row drawn, no warning, one shared time axis,
both files written) and the forecast row by what it draws: the curve over each tile is compared
against the forward's own ``mu_full`` at the anchor ``anchor_index`` names, uncovered raw samples are
gaps, the forecast is drawn in the trace's units, and the shipped geometry renders its dense tiling.
"""
from __future__ import annotations

from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs import warmup_budget as warmup_budget_module  # noqa: E402
from teb_vae.lag_attn_crws import sample_page  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeTrainer  # noqa: E402

from .conftest import (  # noqa: E402
    SHIPPED_SEQUENCE_LENGTH,
    make_stub_batch,
    shipped_warmup_kwargs,
)

#: Raw sampling rate, restated rather than imported: the page's sample-to-second arithmetic is what
#: is under test, and borrowing its own constant would make the assertions circular.
_FS_RAW = 4.0

#: Plausible raw-signal scales, so the context row and the forecast row have something to invert.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: The two rows this cell's borrowed input builder adds, by title prefix.
_INPUT_ROWS = ("Target input", "Source input")

#: Every titled row of this package's page: the sibling's seven plus the two input rows. Stated as
#: arithmetic rather than as ``9`` so a row added to the drawing without one added to the layout --
#: or the reverse -- fails here by name. There is deliberately no third term: this cell reserves no
#: ``forecast_extra_rows`` at all, because its decoder emits $R$ raw samples of one signal and the
#: shipped two-row layout is a picture of exactly that.
_PAGE_ROWS = 7 + len(_INPUT_ROWS)


def _forward(module: Any, data: Any) -> dict:
    """Run the net once and return everything the page builder needs.

    Separate from :func:`_render` because the forward is **stochastic** -- the full branch decodes a
    reparameterised draw -- so a test comparing a drawn artist against the arrays behind it has to
    compare against *this* forward's, not a second one's.

    Args:
        module: A ``SeqVaeLagAttnCrwsTask``.
        data: The batch to run on.

    Returns:
        ``{'outs', 'kld_per_dim', 'target', 'inputs'}``.
    """
    model = module.orig_model
    with torch.no_grad():
        inputs = module._build_forward_inputs(data)
        outs = model(*inputs)
        target, _weight = module._build_raw_target(data)
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    return {"outs": outs, "kld_per_dim": kld_per_dim, "target": target, "inputs": inputs}


def _render(module: Any, data: Any, pieces: Any = None, **overrides) -> Any:
    """Build the whole page for sample 0, through the seams the callback resolves.

    Args:
        module: A ``SeqVaeLagAttnCrwsTask``.
        data: The batch to draw from.
        pieces: A :func:`_forward` result to draw, or ``None`` to run one.
        **overrides: Passed to the builder, e.g. ``normalization_stats=None``.

    Returns:
        The matplotlib ``Figure``. The caller closes it.
    """
    pieces = _forward(module, data) if pieces is None else pieces
    kwargs = dict(
        outs=pieces["outs"],
        kld_per_dim=pieces["kld_per_dim"],
        fhr_raw=pieces["target"],
        geometry=module.orig_model.geometry,
        sample_index=0,
        epoch=3,
        guid="SEG000",
        beta=0.25,
        scalars={"pred_gap": 0.5},
        up_raw=data.up,
        normalization_stats=_STATS,
        # Exactly what ``LagAttnRwsPlotCallback._generate_plots`` passes.
        forecast_rows=module.forecast_rows,
        batch=data,
        input_streams=plotting.input_stream_panels(
            module.orig_model, pieces["inputs"], 0, module.input_stream_panels
        ),
        forecast_extra_rows=tuple(getattr(module, "forecast_extra_rows", ()) or ()),
    )
    kwargs.update(overrides)
    return plotting.build_diagnostic_figure(**kwargs)


def _axes_titled(figure: Any, prefix: str) -> Any:
    """Return the single axes whose title starts with ``prefix``."""
    matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} panel, found {len(matches)}"
    return matches[0]


def _labelled(ax: Any, prefix: str) -> List[Any]:
    """Return the artists on ``ax`` whose legend label starts with ``prefix``."""
    return [line for line in ax.lines if str(line.get_label()).startswith(prefix)]


def _drawn_tiling(pieces: Any, module: Any) -> Any:
    """The anchors, the validity vector and the drawn positions of sample 0.

    Args:
        pieces: A :func:`_forward` result.
        module: The task, for the horizon.

    Returns:
        ``(anchors, valid, positions)``.
    """
    horizon = int(module.orig_model.geometry.horizon)
    anchors = pieces["outs"]["anchor_index"][0].numpy().astype(int)
    valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
    return anchors, valid, sample_page._tiling_anchors(anchors, valid, horizon)


def _trainer_with_batch(batch: Any, **kwargs) -> FakeTrainer:
    """A fake trainer whose validation loader yields ``batch`` once."""
    trainer = FakeTrainer(**kwargs)
    trainer.val_dataloaders = [[batch]]  # type: ignore[attr-defined]
    return trainer


def _warnings_of(function) -> List[str]:
    """Run ``function`` with a loguru sink attached and return the warnings it emitted.

    ``caplog`` cannot see these: loguru does not route through the stdlib ``logging`` module, so a
    ``caplog`` assertion would pass on a callback that warned on every row.

    Args:
        function: A zero-argument callable.

    Returns:
        The warning messages, in order.
    """
    messages: List[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        function()
    finally:
        logger.remove(sink_id)
    return messages


# =================================================================================================
# The callback
# =================================================================================================
def test_the_callback_draws_the_whole_page_and_warns_about_nothing(tmp_path, task, stub_batch, budget):
    """The assertion this file exists for, and it has to be made against the **callback**.

    Every seam is behind a handler that warns and continues, because a figure is never worth failing
    a multi-day fit for -- so a broken row costs one warning and leaves the suite green. Asserted as
    *zero* warnings rather than as the absence of one message, because the handlers emit different
    sentences and any of them is the same defect. Every titled row must carry data on the one shared
    time axis, and the epoch must write both the sample page and the run-level budget figure.
    """
    module = task()
    module.warmup_budget = budget
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=1, file_format="png")
    trainer = _trainer_with_batch(stub_batch)
    figures: List[Any] = []

    original = plotting.build_diagnostic_figure

    def _capture(**kwargs):
        figure = original(**kwargs)
        figures.append(figure)
        return figure

    plotting.build_diagnostic_figure = _capture
    try:
        warnings = _warnings_of(
            lambda: callback._generate_plots(trainer, stub_batch, module, epoch=0)
        )
    finally:
        plotting.build_diagnostic_figure = original

    try:
        assert warnings == []
        titled = [ax for ax in figures[0].axes if ax.get_title()]
        assert len(titled) == _PAGE_ROWS, [ax.get_title()[:30] for ax in titled]
        for prefix in _INPUT_ROWS + ("Forecast", "Raw target FHR"):
            _axes_titled(figures[0], prefix)
        # A column of the page is one instant on every row, the raw-grid forecast row included.
        t_max = stub_batch.fhr.shape[1] / _FS_RAW
        for ax in titled:
            assert ax.has_data(), ax.get_title()
            assert ax.get_xlim() == pytest.approx((0.0, t_max)), ax.get_title()
        # And the two files the epoch produces: the sample page and the run-level budget figure.
        assert (callback.output_dir / f"{warmup_budget_module.BUDGET_FIGURE_STEM}.png").is_file()
        assert list(callback.output_dir.glob("lag_attn_rws_epoch0000_sample0_*.png"))
    finally:
        for figure in figures:
            plt.close(figure)


# =================================================================================================
# The forecast row and the anchor axis
# =================================================================================================
def test_each_drawn_window_carries_the_forecast_of_the_anchor_it_is_drawn_at(task, stub_batch):
    r"""The failure the whole seam exists to prevent, and the one with no exception in it.

    The forecast tensor is $(A_{\max}, H, R)$ indexed by **position in the decoded set**; the raw
    samples a window covers are $[16(t+1),\ 16(t+1) + HR)$ for the *anchor* ``anchor_index`` names
    at that position. The two coincide only at floor $0$ and stride $1$, and the shipped raw rows
    assume exactly that -- so on this model they draw a real forecast a floor's worth of steps
    early, or read past the end of an axis that is $A_{\max}$ long rather than $T_{\mathrm{valid}}$.

    Drawn without normalization statistics, so the comparison is against the forward's own numbers
    rather than against a second application of the loader's affine map.
    """
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces, normalization_stats=None)
    try:
        geometry = module.orig_model.geometry
        anchors, _valid, positions = _drawn_tiling(pieces, module)
        assert len(positions) > 1, "one window would make the placement assertion vacuous"
        block = geometry.horizon * geometry.r

        ax = _axes_titled(figure, "Forecast")
        full_line = _labelled(ax, "full ($z^q$")
        assert len(full_line) == 1
        drawn = np.asarray(full_line[0].get_ydata(), dtype=float)
        assert drawn.size == geometry.raw_len

        for position in positions:
            start = geometry.future_block_start(int(anchors[position]))
            expected = pieces["outs"]["mu_full"][0, position].reshape(-1).numpy()
            assert np.allclose(drawn[start : start + block], expected, atol=1e-5), position
        # And nothing at all is drawn before the first drawn window's own raw block.
        first = geometry.future_block_start(int(anchors[positions[0]]))
        assert np.all(np.isnan(drawn[:first]))
    finally:
        plt.close(figure)


def test_uncovered_raw_samples_are_gaps_rather_than_a_fabricated_continuation(task, stub_batch):
    """The tiling leaves the anchor floor's prefix and whatever tail is not a whole window undrawn.
    Those spans are absent, not predicted, and a line drawn through them would read as a forecast
    the model never made -- on a raw trace, where the eye reads a continuous curve as one signal."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors, _valid, positions = _drawn_tiling(pieces, module)
        first = geometry.future_block_start(int(anchors[positions[0]]))
        last = (
            geometry.future_block_start(int(anchors[positions[-1]]))
            + geometry.horizon * geometry.r
        )
        assert first > 0 and last < geometry.raw_len, "both blank spans must be real spans"

        for label in ("true $Y^{+}", "base ($z^p$", "full ($z^q$"):
            values = np.asarray(_labelled(_axes_titled(figure, "Forecast"), label)[0].get_ydata())
            assert np.all(np.isnan(values[:first])), label
            assert np.all(np.isnan(values[last:])), label
            assert np.isfinite(values[first:last]).all(), label
    finally:
        plt.close(figure)


def test_the_forecast_is_drawn_in_the_same_units_as_the_trace_it_is_read_against(task, stub_batch):
    r"""Both branches and both bands go through the loader's own affine map, because a forecast
    drawn in z-units cannot be checked against physiology by eye -- which is the entire reason the
    normalization statistics are plumbed this far. The truth is drawn once, by the context row that
    owns it, and the forecast row reads that array rather than converting it a second time."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors, _valid, positions = _drawn_tiling(pieces, module)
        block = geometry.horizon * geometry.r
        stats = _STATS["fhr"]

        ax = _axes_titled(figure, "Forecast")
        drawn = np.asarray(_labelled(ax, "base ($z^p$")[0].get_ydata(), dtype=float)
        for position in positions:
            start = geometry.future_block_start(int(anchors[position]))
            expected = (
                pieces["outs"]["mu_base"][0, position].reshape(-1).numpy() * stats["std"]
                + stats["mean"]
            )
            assert np.allclose(drawn[start : start + block], expected, rtol=1e-5), position

        # The truth on the forecast row is the context row's own array, restricted to the drawn
        # windows: one conversion, so the three curves are comparable by construction.
        truth = np.asarray(_labelled(ax, "true $Y^{+}")[0].get_ydata(), dtype=float)
        context = np.asarray(
            _labelled(_axes_titled(figure, "Raw target FHR"), "FHR (bpm)")[0].get_ydata()
        )
        covered = np.isfinite(truth)
        assert covered.any()
        assert np.array_equal(truth[covered], context[covered])
        assert ax.get_ylabel() == "FHR (bpm)"
    finally:
        plt.close(figure)


def test_it_renders_at_the_shipped_geometry(task):
    r"""A page that renders only at the test geometry is not a page -- and the shipped floor is where
    the shipped raw rows do not merely draw at the wrong time but read past the end of the anchor
    axis. At the dense evaluation stride every anchor of $[F, T_{\mathrm{valid}})$ is decoded and
    the drawn windows are the ones a horizon apart from the floor."""
    module = task(model_kwargs=shipped_warmup_kwargs())
    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    pieces = _forward(module, batch)
    figure = _render(module, batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors, valid, positions = _drawn_tiling(pieces, module)

        assert int(valid.sum()) == geometry.t_valid - geometry.warmup
        assert [int(anchors[position]) for position in positions] == list(
            range(geometry.warmup, geometry.t_valid, geometry.horizon)
        )
        ax = _axes_titled(figure, "Forecast")
        assert _labelled(ax, "decoded anchors")[0].get_xdata().size == int(valid.sum())
        assert len([child for child in figure.axes if child.get_title()]) == _PAGE_ROWS
    finally:
        plt.close(figure)
