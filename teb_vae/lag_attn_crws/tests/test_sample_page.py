r"""The row this model replaces, the two it borrows, and the silences that hid all three.

Every builder this cell reaches for exists because a shipped one **fails quietly** on it, and the
three failures are not the same failure.

The forecast rows die *loudly* and are swallowed: the shipped raw rows tile through
``concat_single_forecasts``, which reads its per-anchor block at an **anchor** index, and this
model's forecast is $(A_{\max}, H, R)$ indexed by position in the decoded set. At the shipped
geometry that is $136$ positions read at anchors $134 \dots 269$, so the first read is out of range,
the page builder raises, and the callback's broad handler turns a whole run's diagnostics into one
log line and an empty directory. At a smaller floor the same code does not raise at all -- it draws
a real forecast at the wrong time, at the right shape, in the right colours, on the right axis.

The input rows and the run-level budget figure fail *silently*: both shipped builders consult the
production two-sided Morlet bank, which refuses these channel widths inside handlers that warn and
continue. The symptom of not replacing them is a green suite, two log lines, and a seven-row page.

All three are therefore tested through the **callback**, not only through the builders. A test that
called a replacement directly would pass on a tree where nothing reaches it, which is exactly the
state this package was in before this file existed: every seam was available and the model's
five-tensor forward and tiled anchor axis could not travel down any of them.

The one assertion that fails loudest if the anchor axis is mixed up is the window-placement one: the
drawn curve over each tile is compared against the forward's own ``mu_full`` at the anchor
``anchor_index`` names, so an implementation reading position $k$ as anchor $k$ fails even though
every shape, every axis and every colour is right.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs import sample_page as causal_feature_page  # noqa: E402
from teb_vae.lag_attn_cfs import warmup_budget as warmup_budget_module  # noqa: E402
from teb_vae.lag_attn_cfs.task import SeqVaeLagAttnCfsTask  # noqa: E402
from teb_vae.lag_attn_crws import sample_page  # noqa: E402
from teb_vae.lag_attn_crws.task import SeqVaeLagAttnCrwsTask  # noqa: E402
from teb_vae.lag_attn_rws import input_budget  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws import sample_page as shared_page  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeTrainer  # noqa: E402

from .conftest import (  # noqa: E402
    CAUSAL_C_U,
    CAUSAL_C_Y,
    SHIPPED_SEQUENCE_LENGTH,
    TINY_STRIDE,
    make_stub_batch,
    shipped_warmup_kwargs,
    tiny_warmup_kwargs,
)

#: Raw sampling rate, restated rather than imported: the page's sample-to-second arithmetic is what
#: is under test, and borrowing its own constant would make the assertions circular.
_FS_RAW = 4.0

#: Plausible raw-signal scales, so the context row and the forecast row have something to invert.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: The two rows this cell's borrowed input builder adds, by title prefix.
_INPUT_ROWS = ("Model input — target", "Model input — source")

#: Every titled row of this package's page: the sibling's seven plus the two input rows. Stated as
#: arithmetic rather than as ``9`` so a row added to the drawing without one added to the layout --
#: or the reverse -- fails here by name. There is deliberately no third term: this cell reserves no
#: ``forecast_extra_rows`` at all, because its decoder emits $R$ raw samples of one signal and the
#: shipped two-row layout is a picture of exactly that.
_PAGE_ROWS = 7 + len(_INPUT_ROWS)

#: The four keywords that make a model ungated, as a keyword set the guarded tiny kwargs are
#: overridden with. Written out rather than reached for by name, because "no budget" is four
#: absences and a subset of them is a model the constructor refuses.
_UNGATED = dict(
    target_keep_index=None,
    target_warmup_steps=None,
    source_keep_index=None,
    source_warmup_steps=None,
)


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
# The callback, and the three silences it used to hide
# =================================================================================================
def test_the_callback_draws_the_whole_page_and_warns_about_nothing(tmp_path, task, stub_batch, budget):
    """The assertion this file exists for, and it has to be made against the **callback**.

    Every one of the three seams is behind a handler that warns and continues, because a figure is
    never worth failing a multi-day fit for -- so before this file the run produced a page that was
    never written, two rows that were never drawn, a figure that was never saved, and a green suite.
    Asserted as *zero* warnings rather than as the absence of one message, because the three
    handlers emit three different sentences and any of them is the same defect.
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
        titled = [ax.get_title() for ax in figures[0].axes if ax.get_title()]
        assert len(titled) == _PAGE_ROWS, titled
        for prefix in _INPUT_ROWS + ("Forecast", "Raw target FHR"):
            assert _axes_titled(figures[0], prefix).has_data()
        # And the two files the epoch produces: the sample page and the run-level budget figure.
        assert (callback.output_dir / f"{warmup_budget_module.BUDGET_FIGURE_STEM}.png").is_file()
        assert list(callback.output_dir.glob("lag_attn_rws_epoch0000_sample0_*.png"))
    finally:
        for figure in figures:
            plt.close(figure)


def test_the_two_borrowed_seams_are_the_causal_feature_cells_own_objects(task):
    """Imported rather than rebuilt, and asserted by identity rather than by sampled behaviour.

    These cells read the **same three input tensors** as the causal-feature cells and resolve the
    same warm-up budget -- they differ in what the decoder emits and what the objective scores -- so
    a second implementation of either seam could only differ from the first by being wrong.
    """
    module = task()

    assert (
        SeqVaeLagAttnCrwsTask.__dict__["input_stream_panels"]
        is SeqVaeLagAttnCfsTask.__dict__["input_stream_panels"]
    )
    assert (
        SeqVaeLagAttnCrwsTask.__dict__["input_budget_figure"]
        is SeqVaeLagAttnCfsTask.__dict__["input_budget_figure"]
    )
    # And what the property resolves to on an instance is the sibling's builder, not a wrapper.
    assert module.input_stream_panels is causal_feature_page.causal_stream_panels


def test_the_budget_figure_is_written_once_per_run_under_a_stem_of_its_own(tmp_path, task, budget):
    """A run-level figure is a constant of the configuration rather than of an epoch, and the
    shipped latch is what keeps it that way -- a model this could not describe would otherwise warn
    once per validation epoch for the rest of the fit.

    Its stem is deliberately not the shipped ``causal_input_budget``: the two describe different
    guards -- a two-sided forward reach against a one-sided warm-up -- and a directory holding both
    must be readable rather than ambiguous."""
    module = task()
    module.warmup_budget = budget
    callback = LagAttnRwsPlotCallback(tmp_path, file_format="png")
    trainer = _trainer_with_batch(None)

    callback._write_budget_figure(trainer, module, module.orig_model)
    written = sorted(path.name for path in callback.output_dir.glob("*.png"))
    callback._write_budget_figure(trainer, module, module.orig_model)

    assert callback._budget_figure_written
    assert written == [f"{warmup_budget_module.BUDGET_FIGURE_STEM}.png"]
    assert sorted(path.name for path in callback.output_dir.glob("*.png")) == written
    # The two stems name two different guards and neither may overwrite the other.
    assert warmup_budget_module.BUDGET_FIGURE_STEM != input_budget.BUDGET_FIGURE_STEM


def test_a_task_with_no_budget_refuses_the_figure_by_name(tmp_path, task):
    """The channels the budget **dropped** are the figure's whole subject, and a dropped channel's
    own $W'_c$ is exactly what the checkpoint does not carry -- ``model_kwargs`` stamps the
    survivors' vector, because that is what the constructor needs. So a task that never received a
    budget cannot draw it, and says so by name rather than drawing a figure about the survivors."""
    module = task()
    assert module.warmup_budget is None, "the default must be the absent one"

    with pytest.raises(ValueError, match="no resolved warm-up budget"):
        module.input_budget_figure(tmp_path)


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


def test_the_overlay_draws_the_decoded_anchors_and_the_training_tiling(task, stub_batch):
    r"""Both, because they are two different sets and the page is produced at only one of them.

    Validation decodes every valid anchor at stride $1$, which is what makes the page reproducible;
    training decodes $\{F + \varphi + kS\}$, a fifteenth as many at the shipped geometry, at a phase
    derived per segment per epoch. An overlay showing only the dense set says nothing about the
    geometry the gradients were computed at, which is half of the row's purpose.
    """
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        seconds_per_step = stub_batch.fhr.shape[1] / _FS_RAW / geometry.t
        ax = _axes_titled(figure, "Forecast")
        anchors, valid, _positions = _drawn_tiling(pieces, module)

        rug = _labelled(ax, "decoded anchors")
        assert len(rug) == 1
        assert np.allclose(rug[0].get_xdata(), anchors[valid] * seconds_per_step)

        tiles = _labelled(ax, "training tiles")
        assert len(tiles) == 1, "one legend entry, whatever the tile count"
        # The stride **and** the phase, because the grid drawn is one of $S$ of them and the phase a
        # given segment gets in a given epoch is derived from its own identity: a grid drawn without
        # its phase stated reads as the grid rather than as an example of one.
        assert f"$S$={TINY_STRIDE}" in str(tiles[0].get_label())
        assert "$\\varphi$=0" in str(tiles[0].get_label())
        assert tiles[0].get_xdata()[0] == pytest.approx(geometry.warmup * seconds_per_step)

        floor = _labelled(ax, "anchor floor")
        assert len(floor) == 1
        assert floor[0].get_xdata()[0] == pytest.approx(geometry.warmup * seconds_per_step)
    finally:
        plt.close(figure)


def test_the_overlay_is_read_from_the_forward_rather_than_recomputed(task, stub_batch):
    """So the figure cannot disagree with the loss. Driven by handing the page an anchor set the
    geometry alone would never produce: a recomputing implementation draws the geometry's anchors
    and passes, a reading one draws these."""
    module = task()
    pieces = _forward(module, stub_batch)
    valid = pieces["outs"]["anchor_valid"].clone()
    valid[:, 3:] = False
    pieces["outs"] = dict(pieces["outs"], anchor_valid=valid)

    figure = _render(module, stub_batch, pieces=pieces)
    try:
        ax = _axes_titled(figure, "Forecast")
        rug = _labelled(ax, "decoded anchors")[0]
        assert rug.get_xdata().size == 3
        assert "decoded anchors (3)" in str(rug.get_label())
        # One window survives the truncation, and the row says so rather than reporting the tiling
        # it would have drawn.
        assert "1 of 3 decoded anchors drawn" in ax.get_title()
    finally:
        plt.close(figure)


def test_every_drawn_window_is_marked_at_both_of_its_edges(task, stub_batch):
    """Without the edges the tiling reads as one continuous prediction, which is exactly what it is
    not: each window is decoded from one latent and never sees the window before it. At the
    evaluation resolution consecutive drawn windows abut, so an edge is shared and the count is one
    more than the number of windows."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors, _valid, positions = _drawn_tiling(pieces, module)
        seconds_per_sample = stub_batch.fhr.shape[1] / _FS_RAW / geometry.raw_len
        block = geometry.horizon * geometry.r

        expected = sorted(
            {
                edge
                for position in positions
                for edge in (
                    geometry.future_block_start(int(anchors[position])),
                    geometry.future_block_start(int(anchors[position])) + block,
                )
            }
        )
        ax = _axes_titled(figure, "Forecast")
        # The dashed verticals: the overlay's floor line is solid and labelled and its tile grid is
        # dotted, so the linestyle separates the two sets of verticals on this row.
        drawn = sorted(
            float(np.asarray(line.get_xdata())[0])
            for line in ax.lines
            if np.asarray(line.get_xdata()).size == 2 and line.get_linestyle() == "--"
        )
        assert len(expected) == len(positions) + 1
        assert drawn == pytest.approx([edge * seconds_per_sample for edge in expected])
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


def test_the_page_rows_all_share_one_time_axis(task, stub_batch):
    """Seven inherited and the two input rows; the seam replaces two of the seven and must not touch
    the layout or the axis. A column of the page is one instant on every row, which is what lets a
    reader carry a feature of the forecast down into the lag map -- and it is what a replaced row is
    most likely to break, because the forecast row is the one drawn on the *raw* grid."""
    module = task()
    figure = _render(module, stub_batch)
    try:
        titled = [ax for ax in figure.axes if ax.get_title()]
        assert len(titled) == _PAGE_ROWS, [ax.get_title()[:30] for ax in titled]
        t_max = stub_batch.fhr.shape[1] / _FS_RAW
        for ax in titled:
            assert ax.get_xlim() == pytest.approx((0.0, t_max)), ax.get_title()
            assert ax.has_data(), ax.get_title()
        for prefix in (
            "Target-only latent state",
            "Per-dimension source-conditioned KL",
            "$K_t$",
            "Lag attention",
            r"$\widetilde K_{t,\ell}$",
        ):
            assert _axes_titled(figure, prefix).has_data(), prefix
    finally:
        plt.close(figure)


def test_the_page_carries_the_one_sided_delay_caveat(task, stub_batch):
    r"""One-sidedness and zero latency are different properties and this family buys only the first.
    The forecast claim needs no correction at all here -- an input coefficient at $t$ is a function
    of the past and the target is the raw signal itself -- but a peak at lag $\ell$ is still an
    attribution over stored *input* coefficients, and the correction to a physical delay is
    channel-dependent and of the same order as the lag search.

    The caveat is **one-sided** where the causal-feature page's is two-sided, and that difference is
    the point of this cell: there is no target-side group delay to subtract, because there is no
    target-side filter. Asserted as a string so an edit that keeps the figure rendering cannot drop
    it."""
    figure = _render(task(), stub_batch)
    try:
        assert sample_page.LAG_TIME_CAVEAT in [text.get_text() for text in figure.texts]
        for token in ("stored-coefficient time", "group delay", "one-sided", "$-20$ s"):
            assert token in sample_page.LAG_TIME_CAVEAT, token
        # The two pages say different things, and the sibling's is untouched.
        assert sample_page.LAG_TIME_CAVEAT != causal_feature_page.LAG_TIME_CAVEAT
    finally:
        plt.close(figure)


def test_an_ungated_model_still_draws_every_row(task, stub_batch):
    """Without a budget there is no gate, no warm-up mask and no availability buffer -- and the rows
    are still the rows. A builder that reached for any of them unconditionally would fail on the arm
    the whole family is compared against."""
    module = task(model_kwargs=dict(tiny_warmup_kwargs(anchor_stride=TINY_STRIDE), **_UNGATED))
    figure = _render(module, stub_batch)
    try:
        assert len([ax for ax in figure.axes if ax.get_title()]) == _PAGE_ROWS
        for prefix in _INPUT_ROWS + ("Forecast",):
            assert _axes_titled(figure, prefix).has_data(), prefix
        panels = plotting.input_stream_panels(
            module.orig_model, module._build_forward_inputs(stub_batch), 0,
            module.input_stream_panels,
        )
        assert [panel.values.shape[1] for panel in panels] == [CAUSAL_C_Y, CAUSAL_C_U]
    finally:
        plt.close(figure)


def test_it_renders_at_the_shipped_geometry(task):
    r"""The tiny fixture is a $24$-step window with a floor of $5$; production is $300$ steps, a
    floor of $134$ and $136$ decoded anchors tiled into $5$ windows of $480$ raw samples. A page
    that renders only at the test geometry is not a page -- and the shipped floor is where the
    shipped raw rows do not merely draw at the wrong time but read past the end of the anchor
    axis."""
    module = task(model_kwargs=shipped_warmup_kwargs())
    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    pieces = _forward(module, batch)
    figure = _render(module, batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors, valid, positions = _drawn_tiling(pieces, module)

        assert int(valid.sum()) == geometry.t_valid - geometry.warmup == 136
        assert [int(anchors[position]) for position in positions] == list(
            range(geometry.warmup, geometry.t_valid, geometry.horizon)
        )
        ax = _axes_titled(figure, "Forecast")
        assert f"{len(positions)} of 136 decoded anchors drawn" in ax.get_title()
        assert _labelled(ax, "decoded anchors")[0].get_xdata().size == 136
        assert len([child for child in figure.axes if child.get_title()]) == _PAGE_ROWS
    finally:
        plt.close(figure)


# =================================================================================================
# What is reached rather than copied
# =================================================================================================
def test_the_anchor_walk_and_the_overlay_are_the_siblings_own_functions():
    """Both take an anchor set, a validity vector and a horizon and name no channel, no coefficient
    and no target at all, so a copy of either here could only drift from the tiling the sibling
    draws -- and the two pages would then disagree about which anchors a run decoded."""
    assert sample_page._tiling_anchors is causal_feature_page._tiling_anchors
    assert sample_page._draw_anchor_overlay is causal_feature_page._draw_anchor_overlay
    assert sample_page.raw_context_row is shared_page.raw_context_row
    assert sample_page.BAND_SIGMAS is shared_page.BAND_SIGMAS


def test_the_shared_layout_is_reached_rather_than_copied():
    """No ``lag_attn_crws/plotting.py`` and no callback of this package's own: the seams exist so a
    sibling supplies rows and inherits the rest, and a second callback class would be a second place
    for the layout, the cuts and the caption to drift.

    The drawing module builds no figure and no GridSpec either -- it draws through ``row_axes`` and
    ``finalise_time_axis``, which are what make a column of the page one instant on every row. An
    implementation that set its own limits would break exactly the alignment the figure is read
    by."""
    package = Path(sample_page.__file__).parent

    assert not (package / "plotting.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("teb_vae.lag_attn_crws.plotting")

    # The layout tools are not merely unused, they are unreachable: a module that cannot name
    # ``pyplot`` or a ``GridSpec`` cannot open a second figure or a second row list.
    for tool in ("plt", "GridSpec", "build_diagnostic_figure"):
        assert not hasattr(sample_page, tool), tool

    source = inspect.getsource(sample_page)
    for forbidden in ("add_subplot", "set_xlim"):
        assert forbidden not in source, forbidden
    for reused in ("rows.row_axes(", "rows.finalise_time_axis("):
        assert reused in source, reused
