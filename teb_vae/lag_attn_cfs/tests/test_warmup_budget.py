r"""The two figures the warm-up budget is read through: the run's own, and the curve that chose it.

They are drawn from the same resolved object and answer different questions, and the split is not
cosmetic. The run-level figure is a constant of the *configuration* -- which channels this run kept
and how long each waits -- so the diagnostic callback writes it once per run beside the pages. The
tradeoff curve is a constant of the *shard*: every candidate budget's channels, anchors and tiles,
computed from the same vectors, and identical in every run against those shards. Writing it per run
would be waste, so it is produced offline from an entry point that needs no fit.

Two things about the run-level figure are worth stating because they look like details and are the
reason it exists at all. It is drawn from the **declared**-width warm-up vectors, so it can show the
channels the budget dropped beside the ones it kept -- and those are precisely what a checkpoint
cannot supply, since ``model_kwargs`` stamps the survivors' vector alone. And it reuses the shipped
budget panel by object identity while replacing the figure-level function around it, because that
function calls ``describe_streams``, which builds the production two-sided Morlet bank and refuses
these channel widths.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn_cfs import causal_warmup, warmup_budget  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget  # noqa: E402
from teb_vae.lag_attn_rws import input_budget, plotting  # noqa: E402
from teb_vae.lag_attn_rws import sample_page as shared_page  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeTrainer  # noqa: E402

from .conftest import (  # noqa: E402
    SHIPPED_BUDGET_STEPS,
    SHIPPED_HORIZON,
    SHIPPED_SEQUENCE_LENGTH,
    causal_config,
    make_stub_batch,
    shipped_warmup_kwargs,
)

#: Seconds per decimated step, restated rather than imported: the figures' step-to-seconds
#: arithmetic is what is under test, and borrowing their own constant would make it circular.
_STEP_S = 4.0

#: What the shipped budget resolves to on the committed fixture. Derived in the tests from the
#: resolver wherever it is asserted; written here only for the "this fixture is the one we think"
#: guard, which a rebuild at another quantile should fail rather than quietly pass.
_KEPT_TARGET = 98
_DECLARED_TARGET = 102


@pytest.fixture
def shipped_budget():
    """The resolved warm-up budget at the shipped threshold, against the committed fixture."""
    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return resolved


def _bars(ax: Any) -> List[Any]:
    """The panel's bar containers, kept first then dropped, as the shipped panel draws them."""
    return list(ax.containers)


def _warnings_of(function) -> List[str]:
    """Run ``function`` with a loguru sink attached and return the warnings it emitted."""
    messages: List[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        function()
    finally:
        logger.remove(sink_id)
    return messages


# =================================================================================================
# The run-level figure
# =================================================================================================
def test_the_shared_panel_and_annotation_are_reused_rather_than_reimplemented():
    """By object identity, so the two figures cannot come to draw a bar differently. The
    figure-level function around it is **not** reusable: it calls ``describe_streams``, which
    builds the production Morlet bank and refuses these channel widths."""
    assert warmup_budget._budget_panel is input_budget._budget_panel
    assert input_budget.annotate_channel_frequencies is shared_page.annotate_channel_frequencies
    assert not hasattr(warmup_budget, "describe_streams")
    assert warmup_budget.BUDGET_FIGURE_STEM != input_budget.BUDGET_FIGURE_STEM


def test_each_kept_bar_spans_minus_delta_warmup_to_zero(shipped_budget):
    r"""What feeding $\delta = W'$ and $\rho = \Delta W'$ into the shipped panel buys: a bar that
    is a **backward settling length**, ending exactly at the anchor's causal endpoint. The reading
    is the mirror image of the two-sided figure's, where a bar ending after $0$ reaches into the
    window it is meant to forecast."""
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        kept, dropped = _bars(figure.axes[0])
        warmup = np.asarray(shipped_budget.target.warmup_steps, dtype=float)

        assert len(kept.patches) == shipped_budget.target.kept_width == _KEPT_TARGET
        assert np.allclose(
            sorted(patch.get_x() for patch in kept.patches),
            sorted(-_STEP_S * warmup),
        )
        assert np.allclose([patch.get_x() + patch.get_width() for patch in kept.patches], 0.0)
        # And the dropped ones are drawn forward from the anchor instead, which is what makes them
        # legible: their bar runs through the window they were still warming up for.
        assert len(dropped.patches) == _DECLARED_TARGET - _KEPT_TARGET
        assert all(patch.get_x() == pytest.approx(0.0) for patch in dropped.patches)
        assert all(patch.get_width() > SHIPPED_HORIZON * _STEP_S for patch in dropped.patches)
    finally:
        plt.close(figure)


def test_the_budget_is_marked_and_the_forecast_window_is_shaded(shipped_budget):
    r"""$-\Delta B = -536$ s at the shipped budget, and $[0, +60]$ s at the shipped horizon. Both
    are on the axis the bars are drawn against, so "no kept target bar starts left of the budget"
    is checkable by eye rather than by measuring."""
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        expected = -_STEP_S * shipped_budget.target.max_warmup
        assert expected == pytest.approx(-536.0)
        for ax in figure.axes:
            marks = [
                line for line in ax.lines
                if str(line.get_label()).startswith("budget")
            ]
            assert len(marks) == 1
            assert marks[0].get_xdata()[0] == pytest.approx(expected)

        window = [
            (patch.get_x(), patch.get_x() + patch.get_width())
            for patch in figure.axes[0].patches
            if patch.get_width() == pytest.approx(SHIPPED_HORIZON * _STEP_S)
        ]
        assert window == [pytest.approx((0.0, 60.0))]
        # The whole point of the mark: every kept target bar starts at or right of it.
        kept, _dropped = _bars(figure.axes[0])
        assert min(patch.get_x() for patch in kept.patches) >= expected
    finally:
        plt.close(figure)


def test_no_kept_bar_is_cut_by_the_left_edge(shipped_budget):
    """The panel counts what runs past the *right* edge and reports it in the caption; a bar cut on
    the left would be reported by nothing and would read as though it began at the axis. The source
    is never gated and its slowest kept channel waits twice as long as the target's, so a limit
    taken from the budget alone would do exactly that."""
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        for ax, stream in zip(figure.axes, (shipped_budget.target, shipped_budget.source)):
            left, _right = ax.get_xlim()
            assert left <= -_STEP_S * stream.max_warmup
            kept = _bars(ax)[0]
            assert min(patch.get_x() for patch in kept.patches) >= left
        assert shipped_budget.source.max_warmup > shipped_budget.target.max_warmup, (
            "an ungated source that waits no longer than the target makes this vacuous"
        )
    finally:
        plt.close(figure)


def test_the_panel_titles_name_a_warm_up_rather_than_a_delay(shipped_budget):
    """The shared panel titles what it draws a *delay*, which is what it is on the two-sided figure
    and is not what it is here: nothing is shifted, and the region behind the boundary holds real
    values on no defined scale rather than a zero fill."""
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        target, source = (ax.get_title() for ax in figure.axes[:2])
        assert "warm-up 0-134 steps (0-536 s)" in target
        assert "delay" not in target and "delay" not in source
        assert f"{_KEPT_TARGET}/{_DECLARED_TARGET} channels kept" in target
        assert "fhr_st 32/36, fhr_ph 66/66" in target
        assert "up_st 36/36, up_ph 15/15" in source
    finally:
        plt.close(figure)


def test_the_block_dividers_are_in_declared_channel_coordinates(shipped_budget):
    """Unlike the per-sample input rows, which are in surviving-channel coordinates. This figure's
    y-axis *is* the declared channel -- it has to be, since it draws the channels the budget removed
    -- so the boundary belongs where the declared widths put it, and a divider mapped to surviving
    coordinates would name the wrong channel on the one figure where both kinds exist."""
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        split = shipped_budget.target.block_spans[1][1]
        dividers = [
            line.get_ydata()[0] for line in figure.axes[0].lines
            if line.get_linestyle() == "--" and not str(line.get_label()).startswith("budget")
        ]
        assert dividers == pytest.approx([split - 0.5])
        assert split == 36
    finally:
        plt.close(figure)


def test_the_callback_writes_it_once_per_run_under_its_own_stem(tmp_path, task):
    """Behind the existing latch, which is set *before* the attempt so a model it cannot describe
    warns once rather than once per validation epoch. The stem is distinct from the shipped
    ``causal_input_budget``, so a directory holding both is readable rather than ambiguous."""
    module = task(model_kwargs=shipped_warmup_kwargs())
    module.warmup_budget = resolve_warmup_budget(causal_config())
    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=1, file_format="png")
    trainer = FakeTrainer()
    trainer.val_dataloaders = [[batch]]  # type: ignore[attr-defined]

    warnings = _warnings_of(
        lambda: [
            callback._generate_plots(trainer, batch, module, epoch=epoch)
            for epoch in (0, 1)
        ]
    )
    plt.close("all")

    written = sorted(path.name for path in callback.output_dir.iterdir() if path.is_file())
    assert warnings == []
    assert callback._budget_figure_written is True
    assert f"{warmup_budget.BUDGET_FIGURE_STEM}.png" in written
    assert f"{input_budget.BUDGET_FIGURE_STEM}.png" not in written
    # One figure for two epochs, and the pages for both.
    assert len([name for name in written if name.startswith("causal_")]) == 1
    assert len([name for name in written if name.startswith("lag_attn_rws_epoch")]) == 2


def test_a_task_without_a_resolved_budget_costs_the_figure_and_nothing_else(
    tmp_path, task, stub_batch
):
    """The seam is a method rather than a property for exactly this: the callback resolves it with
    ``getattr(..., None)``, which does not swallow an exception raised inside a property, so a
    raising property would take down the whole page instead of the one figure it cannot draw."""
    module = task()
    assert module.warmup_budget is None
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=1, file_format="png")
    trainer = FakeTrainer()
    trainer.val_dataloaders = [[stub_batch]]  # type: ignore[attr-defined]

    warnings = _warnings_of(
        lambda: callback._generate_plots(trainer, stub_batch, module, epoch=0)
    )
    plt.close("all")

    assert len([message for message in warnings if "input-budget figure skipped" in message]) == 1
    assert [message for message in warnings if "input rows skipped" in message] == []
    assert list(callback.output_dir.glob("lag_attn_rws_epoch*.png"))


# =================================================================================================
# The offline tradeoff curve
# =================================================================================================
def test_the_shipped_budget_reads_98_channels_152_anchors_and_11_tiles(shipped_budget):
    r"""The row of the tradeoff that justifies the shipped floor, asserted against the resolver
    rather than against literals -- so a fixture rebuilt at another ``causal_warmup_quantile``
    re-derives it instead of failing on a stale constant.

    The floor comes from the **survivors'** own maximum, not from the threshold: a budget of $151$
    keeps the identical channels whose slowest still waits $134$ steps, so a floor read off the
    threshold would sit $17$ steps too high and cost two tiles for nothing.
    """
    points = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
    )
    at_budget = [point for point in points if point.budget_steps == SHIPPED_BUDGET_STEPS]

    assert len(at_budget) == 1
    point = at_budget[0]
    t_valid = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON
    assert point.kept == shipped_budget.target.kept_width
    assert point.anchors == t_valid - (shipped_budget.target.max_warmup - 1)
    assert point.tiles == -(-point.anchors // SHIPPED_HORIZON)
    assert (point.kept, point.anchors, point.tiles) == (98, 152, 11)


def test_a_threshold_above_the_staircase_buys_nothing(shipped_budget):
    """Which is the whole argument for the shipped floor, and it is a property of the data rather
    than of the threshold: between two steps of the staircase the kept set does not move, so
    neither does the floor the survivors force."""
    points = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
        thresholds=[SHIPPED_BUDGET_STEPS, SHIPPED_BUDGET_STEPS + 17],
    )

    assert points[0][1:] == points[1][1:]


def test_the_curve_is_computed_from_the_resolved_vectors_not_from_constants(shipped_budget):
    """A dataset rebuilt at another quantile changes both the warm-up vectors and the stored
    channel count, so a curve drawn from literals would describe a boundary the data no longer
    has. Driven by moving the vector rather than by inspecting the code."""
    slowed = dataclasses.replace(
        shipped_budget.target,
        declared_warmup_steps=tuple(
            step + 40 for step in shipped_budget.target.declared_warmup_steps
        ),
    )
    common = dict(
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
        thresholds=[SHIPPED_BUDGET_STEPS],
    )

    original = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps, **common
    )
    moved = warmup_budget.budget_tradeoff(slowed.declared_warmup_steps, **common)

    # Fewer survive the same threshold -- and the anchors go *up*, not down, because the survivors
    # are then the fast channels alone and the floor they force is lower. Both directions come
    # from the vector rather than from the threshold, which is the property under test.
    assert moved[0].kept < original[0].kept
    assert moved[0].anchors > original[0].anchors


def test_the_infeasible_region_is_where_no_tile_fits(shipped_budget):
    r"""At the shipped one-minute horizon every threshold still admits a tile, and at two minutes
    the slowest channels do not: keeping all $102$ is then *geometrically impossible*, because
    $T_{\mathrm{valid}}$ is shorter by a whole horizon. That is what makes the horizon a lever on
    the warm-up cost and not merely a forecast length, so the figure shades it rather than letting
    the curves run to zero unremarked."""
    at_one_minute = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
    )
    at_two_minutes = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=2 * SHIPPED_HORIZON,
        anchor_stride=2 * SHIPPED_HORIZON,
    )

    assert all(point.tiles >= 1 for point in at_one_minute)
    assert at_two_minutes[-1].tiles == 0

    figure = warmup_budget.build_tradeoff_figure(at_two_minutes, shipped_budget=134)
    try:
        shaded = [
            patch for patch in figure.axes[0].patches
            if str(patch.get_label()) == "no tile fits"
        ]
        assert len(shaded) == 1
        first_infeasible = min(
            point.budget_steps for point in at_two_minutes if point.tiles == 0
        )
        assert shaded[0].get_x() == pytest.approx(first_infeasible - 0.5)
    finally:
        plt.close(figure)


def test_the_figure_draws_three_curves_a_seconds_twin_and_the_shipped_mark(shipped_budget):
    """The three quantities move against each other and the choice is where they cross, so all
    three are on one axis; the twin is there because the budget is argued about in minutes of
    recording and read off a config in steps."""
    points = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
    )
    figure = warmup_budget.build_tradeoff_figure(points, shipped_budget=SHIPPED_BUDGET_STEPS)
    # Matplotlib defers a secondary axis's limits to draw time, so an assertion made before one
    # passes against the default $(0, 1)$ whatever the transform is.
    figure.canvas.draw()
    try:
        ax = figure.axes[0]
        labels = [str(line.get_label()) for line in ax.lines]
        assert [label for label in labels if not label.startswith("_")] == [
            "target channels kept",
            "anchors admitted",
            "tiles per sample at $\\varphi=0$",
        ]
        assert len(ax.child_axes) == 1
        low, high = ax.get_xlim()
        assert ax.child_axes[0].get_xlim() == pytest.approx((_STEP_S * low, _STEP_S * high))
        marks = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(marks) == 1 and marks[0].get_xdata()[0] == SHIPPED_BUDGET_STEPS
        assert f"$B$={SHIPPED_BUDGET_STEPS}: 98 ch, 152 anchors, 11 tiles" in [
            text.get_text() for text in ax.texts
        ]
    finally:
        plt.close(figure)


def test_the_offline_entry_point_runs_with_no_command_line(tmp_path, monkeypatch):
    """The Run-button convention: a module-level constant naming the config, no ``required=True``
    anywhere to fire before it is read, and ``main`` returning the exit code rather than calling
    ``sys.exit`` -- so this test can call it directly."""
    repo_root = Path(causal_warmup.__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(causal_warmup, "RUN_OUTPUT_DIR", str(tmp_path))

    assert isinstance(causal_warmup.RUN_CONFIG, str)
    assert causal_warmup._cli() == 0
    assert (tmp_path / f"{warmup_budget.TRADEOFF_FIGURE_STEM}.pdf").exists()

    source = Path(causal_warmup.__file__).read_text(encoding="utf-8")
    assert "required=True" not in source
    assert "sys.exit(_cli())" in source


def test_the_entry_point_refuses_a_config_with_no_budget_by_name(tmp_path, monkeypatch):
    """Naming the constant to edit, because a Run-button launch has no command line for the message
    to point at."""
    repo_root = Path(causal_warmup.__file__).resolve().parents[2]
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(causal_warmup, "RUN_CONFIG", str(tmp_path / "absent.yaml"))

    assert causal_warmup._cli() == 2
