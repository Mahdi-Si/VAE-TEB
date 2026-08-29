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
        # H * 4 s = 120 s at the shipped two-minute horizon.
        assert window == [pytest.approx((0.0, 120.0))]
        # The whole point of the mark: every kept target bar starts at or right of it.
        kept, _dropped = _bars(figure.axes[0])
        assert min(patch.get_x() for patch in kept.patches) >= expected
    finally:
        plt.close(figure)


def test_no_kept_bar_is_cut_by_the_left_edge(shipped_budget):
    """The panel counts what runs past the *right* edge and reports it in the caption; a bar cut on
    the left would be reported by nothing and would read as though it began at the axis.

    The guard against vacuity is that a kept bar actually **reaches** the stream's maximum: without
    it a limit taken from anywhere at all would pass. It used to be "the source is ungated and its
    slowest kept channel waits twice as long as the target's", and the alignment retired that -- the
    reference drops every source channel above it, so the two streams' kept maxima are now equal, at
    the budget, which is the zero-marginal-warm-up lemma showing up in a figure.
    """
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        for ax, stream in zip(figure.axes, (shipped_budget.target, shipped_budget.source)):
            left, _right = ax.get_xlim()
            assert left <= -_STEP_S * stream.max_warmup
            kept = _bars(ax)[0]
            longest = min(patch.get_x() for patch in kept.patches)
            assert longest == pytest.approx(-_STEP_S * stream.max_warmup), stream.name
            assert longest >= left, stream.name
        assert (
            shipped_budget.source.max_warmup
            == shipped_budget.target.max_warmup
            == SHIPPED_BUDGET_STEPS
        )
    finally:
        plt.close(figure)


def test_the_panel_titles_name_a_warm_up_rather_than_a_delay(shipped_budget):
    """The shared panel titles what it draws a *delay*, which is what it is on the two-sided figure
    and is not what it is here: what these bars measure is a settling length, and the region behind
    the boundary holds real values on no defined scale rather than a zero fill.

    The alignment does introduce a genuine per-channel delay, and this figure deliberately does not
    draw it: the shift is a separate vector on the resolved budget, so a bar that silently became
    $W'_c + d_c$ would make the two figures of this family measure different quantities under one
    caption. The titles must therefore keep saying warm-up.
    """
    figure = warmup_budget.build_warmup_budget_figure(shipped_budget, horizon=SHIPPED_HORIZON)
    try:
        target, source = (ax.get_title() for ax in figure.axes[:2])
        assert "warm-up 0-134 steps (0-536 s)" in target
        assert "delay" not in target and "delay" not in source
        assert f"{_KEPT_TARGET}/{_DECLARED_TARGET} channels kept" in target
        assert "fhr_st 32/36, fhr_ph 66/66" in target
        # The aligned source: the reference removes the four `up_st` channels above it and leaves
        # every one of the fifteen `up_ph`, which is why that reference was chosen.
        assert "up_st 32/36, up_ph 15/15" in source
        assert "47/51 channels kept" in source
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
    assert (point.kept, point.anchors, point.tiles) == (98, 137, 5)


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
    r"""**The infeasible region exists at the shipped horizon**, and that is the whole point of
    shading it.

    At one minute every threshold still admits a tile. At the shipped two minutes the slowest ones
    do not: keeping all $102$ channels needs a floor of $277$ against $T_{\mathrm{valid}} = 270$, so
    it is *geometrically impossible* rather than merely expensive. $T_{\mathrm{valid}}$ is shorter
    by a whole horizon, which is what makes the horizon a lever on the warm-up cost and not merely a
    forecast length -- and it is why the figure shades the region rather than letting the curves run
    to zero unremarked.

    This assertion **reversed** when the shipped horizon moved from $15$ to $30$: the shaded region
    used to be reachable only by an arm, and now the shipped configuration sits in the same figure
    as it. The shipped budget of $134$ is far from that edge, so nothing about the shipped run
    changes; what changes is that the figure's warning is about the configuration a reader is
    looking at rather than about a hypothetical one.
    """
    at_one_minute = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON // 2,
        anchor_stride=SHIPPED_HORIZON // 2,
    )
    at_two_minutes = warmup_budget.budget_tradeoff(
        shipped_budget.target.declared_warmup_steps,
        sequence_length=SHIPPED_SEQUENCE_LENGTH,
        horizon=SHIPPED_HORIZON,
        anchor_stride=SHIPPED_HORIZON,
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
        assert f"$B$={SHIPPED_BUDGET_STEPS}: 98 ch, 137 anchors, 5 tiles" in [
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


# =================================================================================================
# The identifiability check's own lag geometry, which the tradeoff below is priced at.
#
# Written out rather than loaded from `configs/planted.yaml`: reading both from the config the check
# also reads would make the band arithmetic agree with itself. These are the two leaves that config
# pins and refuses to retune, so they are stated here as the claim.
# =================================================================================================
_PLANTED_HORIZON = 30
_PLANTED_MAX_LAG = 90


# =================================================================================================
# The source-reference tradeoff
#
# The measurement that priced the shipped source clock, and the properties that make its table
# readable. Three quantities move against each other -- how many source channels survive, how fresh
# the freshest survivor is, and where a physiological delay lands on the lag axis -- and the pin was
# chosen where they cross. What is checked here is not the pin (a number is a decision, and it lives
# in the config) but that the table computing it says what it claims to say.
#
# Every one of these could be wrong in a way a table still prints. A candidate between two stored
# delays is a clock no channel keeps. A "kept" count that included a channel above the reference
# would be counting a shift that reads the channel's own future. And the band arithmetic is an
# inverse of an identity with three constants in it, where a sign error moves the whole answer and
# nothing looks odd.
# =================================================================================================
@pytest.fixture(scope="module")
def source_points():
    """The tradeoff over the shipped source stream, at the identifiability check's lag geometry."""
    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return warmup_budget.source_reference_tradeoff(
        resolved.source,
        target_reference_s=float(resolved.reference_delay_s),
        horizon=_PLANTED_HORIZON,
        max_lag=_PLANTED_MAX_LAG,
    )


#: The shipped source stream's own stored delays, one per DECLARED channel. Every candidate has to
#: be one of these: the resolver snaps an explicit float against them and refuses one that matches
#: none, so a table offering an unsnappable candidate would be offering a value the config cannot
#: carry -- and the operator would find that out at launch rather than here.
def _stored_source_delays():
    resolved = resolve_warmup_budget(causal_config())
    assert resolved is not None
    return [float(delay) for delay in resolved.source.declared_delay_s]


def test_every_candidate_is_a_delay_some_source_channel_actually_keeps(source_points):
    """A reference between two stored delays is a clock no channel reports on."""
    delays = {round(delay, 4) for delay in _stored_source_delays()}

    assert source_points, "the tradeoff produced no candidates at all"
    for point in source_points:
        assert round(point.reference_s, 4) in delays, point.reference_s


def test_the_candidates_are_ordered_and_span_the_floor_to_the_targets_own_clock(source_points):
    """Ordered, because the table is read down a column; and bounded above by the target's own
    reference, because a source clock slower than the target's buys nothing the single-reference
    scheme did not already have."""
    references = [point.reference_s for point in source_points]
    resolved = resolve_warmup_budget(causal_config())

    assert references == sorted(references)
    assert references[0] >= warmup_budget.CANDIDATE_FLOOR_SECONDS
    assert references[-1] == pytest.approx(float(resolved.reference_delay_s))


def test_a_faster_clock_keeps_fewer_channels_and_a_fresher_one(source_points):
    """The trade itself, as a monotone property rather than as two numbers.

    This is the whole content of the measurement: a shift can only delay, so the reference is
    bounded below by the slowest channel it keeps, and buying recency means dropping channels.
    A table where the two moved together would be describing a free lunch, which is the shape a
    sign error takes here.
    """
    for slower, faster in zip(source_points[1:], source_points):
        assert faster.reference_s < slower.reference_s
        assert faster.kept <= slower.kept
        assert faster.recency_s < slower.recency_s


def test_the_kept_count_is_the_channels_at_or_below_the_reference(source_points):
    """Counted against the stream's own delays rather than taken from the point, because "kept" is
    the whole cost side of the trade: a count that admitted a channel above the reference would be
    counting a shift that reads that channel's own future, which is the property the alignment
    exists to preserve."""
    delays = _stored_source_delays()

    for point in source_points:
        expected = sum(1 for delay in delays if float(delay) <= point.reference_s + 1e-6)
        assert point.kept == expected, point.reference_s
        assert point.declared == len(delays)
        assert sum(kept for _, kept, _ in point.block_counts) == point.kept


def test_the_envelope_survivors_are_the_criterion_the_pin_was_taken_on(source_points):
    """The decision criterion is stated over one block, and the point carries that block's count
    beside the total for exactly that reason: a fast clock that kept a healthy total while pricing
    the contraction envelope out would improve the lag axis by discarding the physiological signal
    path, which is the failure the criterion exists to refuse."""
    for point in source_points:
        by_name = {name: kept for name, kept, _ in point.block_counts}
        assert point.envelope_kept == by_name[warmup_budget.ENVELOPE_BLOCK]
        assert point.meets_envelope_criterion == (
            point.envelope_kept >= warmup_budget.MIN_UP_PH_KEPT
        )


def test_the_offset_is_the_two_clocks_difference_and_is_zero_at_the_targets_own(source_points):
    r"""$\tau^u_{\mathrm{ref}} - \tau^y_{\mathrm{ref}}$, which is the single constant a dual
    reference puts on the lag axis in place of the unaligned arm's pair-indexed smear. It is
    negative for every faster candidate and exactly zero at the target's own clock, which is the
    single-reference scheme -- so the last row of the table is the arm the revision moved away
    from, priced beside the ones it moved to."""
    resolved = resolve_warmup_budget(causal_config())
    target = float(resolved.reference_delay_s)

    for point in source_points:
        assert point.offset_s == pytest.approx(point.reference_s - target, abs=1e-6)
        assert point.offset_s <= 1e-6
    assert source_points[-1].offset_s == pytest.approx(0.0, abs=1e-6)


def test_the_band_lags_are_the_physical_lag_identity_solved_for_the_lag(source_points):
    r"""The arithmetic, against the identity written out here rather than against the function that
    computes it.

    $\ell = (\tau^{\mathrm{phys}} + \tau_{\mathrm{pre}} - \mathrm{offset}) / \Delta - 1 - h$. The
    low end is the *fastest* delay at the *furthest* horizon step and the high end is the slowest
    at the first, because $\ell$ falls with $h$ -- getting that pairing backwards would report a
    band that clears both edges when it does not, and the whole reference decision rests on which
    edges it clears.
    """
    lo_s, hi_s = warmup_budget.PHYSIOLOGICAL_BAND_SECONDS
    delta = warmup_budget.SECONDS_PER_STEP
    pre = warmup_budget.MECHANICAL_SHIFT_SECONDS

    for point in source_points:
        expected_lo = (lo_s + pre - point.offset_s) / delta - 1.0 - (_PLANTED_HORIZON - 1)
        expected_hi = (hi_s + pre - point.offset_s) / delta - 1.0 - 0.0
        assert point.band_lag_lo == pytest.approx(expected_lo, abs=1e-6)
        assert point.band_lag_hi == pytest.approx(expected_hi, abs=1e-6)
        assert point.readable == (0.0 <= point.band_lag_lo and point.band_lag_hi <= _PLANTED_MAX_LAG)


def test_the_targets_own_clock_censors_the_band_and_a_faster_one_does_not(source_points):
    """The finding the whole measurement was taken for, as a property of the table rather than as a
    recorded number: at the single reference the physiological band falls below lag 0 for most of
    the horizon, and some faster candidate clears both edges at every horizon step. A table on
    which no candidate was readable would price nothing."""
    assert not source_points[-1].readable, "the single-reference arm is not censored"
    assert any(point.readable for point in source_points)
    # And the readable ones are the fast end: readability is monotone in the clock, so a readable
    # candidate slower than an unreadable one would mean the band arithmetic is not monotone in the
    # offset -- which it is, by inspection of the identity above.
    readable = [point.reference_s for point in source_points if point.readable]
    unreadable = [point.reference_s for point in source_points if not point.readable]
    assert max(readable) < min(unreadable)


def test_the_table_prints_every_candidate_with_its_verdict(source_points):
    """The operator-facing half. The table is what the pin was read off, so every row has to carry
    the three quantities and a verdict -- a table that printed only the winner would leave the
    decision unreproducible from its own output."""
    table = warmup_budget.format_source_reference_table(
        source_points, max_lag=_PLANTED_MAX_LAG, horizon=_PLANTED_HORIZON
    )

    assert table.count("\n") >= len(source_points)
    for point in source_points:
        assert f"{point.reference_s:.4f}" in table
    assert "CENSORED" in table and "readable" in table
