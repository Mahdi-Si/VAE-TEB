r"""The three rows this model replaces, and the two silences that made replacing them necessary.

Both builders this package supplies exist because the shipped ones **fail quietly** on it. The
input rows are assembled by a function welded to the production two-sided Morlet bank, which
refuses these channel widths inside a handler that warns and continues -- so the symptom of not
replacing it is a green suite, one log line, and a page with two rows missing. The forecast rows
walk a dense $(T_{\mathrm{valid}}, H, C)$ block and index an *anchor* into its first axis, while
this model's forecast is $(A_{\max}, H, C)$ indexed by position in the decoded set -- so the
symptom there is worse: a real forecast drawn at the wrong time, with no exception anywhere in it.

Both are therefore tested through the **callback**, not only through the builders. A test that
called the replacement directly would pass on a tree where nothing reaches it, which is exactly the
state this package was in before: the seam existed and the model's five-tensor forward could not
travel down it.

The one assertion that fails loudest if the anchor axis is mixed up is the window-placement one:
the drawn curve at the first tile is compared against the forward's own ``mu_full`` at the anchor
``anchor_index`` names, so an implementation reading position $k$ as anchor $k$ fails even though
every shape, every axis and every colour is right.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from teb_vae.lag_attn import figure_primitives  # noqa: E402
from teb_vae.lag_attn_cfs import sample_page  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import SOURCE_BLOCKS  # noqa: E402
from teb_vae.lag_attn_fs import sample_page as feature_page  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws import sample_page as shared_page  # noqa: E402
from teb_vae.lag_attn_rws.nets import losses, raw_masks  # noqa: E402
from teb_vae.lag_attn_rws.plotting import LagAttnRwsPlotCallback  # noqa: E402
from train.test_utils import FakeTrainer  # noqa: E402

from .conftest import (  # noqa: E402
    CAUSAL_C_U,
    CAUSAL_C_Y,
    TINY_STRIDE,
    make_stub_batch,
    tiny_warmup_kwargs,
)

#: Raw sampling rate, restated rather than imported: the page's second-to-time arithmetic is what
#: is under test, and borrowing its own constant would make the assertions circular.
_FS_RAW = 4.0

#: Plausible raw-signal scales, so the context row has something to invert. Only the two raw
#: signals carry statistics; the forecast target is the loader's ``normalize_fields`` output used
#: as delivered.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: The two rows this package's input builder adds, by title prefix.
_INPUT_ROWS = ("Model input — target", "Model input — source")

#: Every titled row of this package's page: the sibling's seven, the two input rows, and the six
#: of ``CAUSAL_EXTRA_ROWS``. Stated as arithmetic rather than as ``15`` so a row added to the
#: drawing module without one added to the layout -- or the reverse -- fails here by name.
_PAGE_ROWS = 7 + len(_INPUT_ROWS) + len(sample_page.CAUSAL_EXTRA_ROWS)


def _forward(module: Any, data: Any) -> dict:
    """Run the net once and return everything the page builder needs.

    Separate from :func:`_render` because the forward is **stochastic** -- the full branch decodes
    a reparameterised draw -- so a test comparing a drawn artist against the arrays behind it has
    to compare against *this* forward's, not a second one's.

    Args:
        module: A ``SeqVaeLagAttnCfsTask``.
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
        module: A ``SeqVaeLagAttnCfsTask``.
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
        # The builder's parameter is still named for the raw model's target; what this model
        # passes through it is the concatenated feature stream its loss was computed against.
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
        forecast_extra_rows=module.forecast_extra_rows,
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
# The input rows, and the silence that hid their absence
# =================================================================================================
def test_the_callback_draws_both_input_rows_and_warns_about_nothing(tmp_path, task, stub_batch):
    """The assertion this task exists for, and it has to be made against the **callback**.

    ``input_stream_panels`` wraps its builder in a broad ``except Exception`` that warns and
    returns no rows, because the seven rows below do not depend on them -- so a page missing both
    of them is indistinguishable from a page that never wanted them unless the log is asserted
    too. Before the replacement this run produced three swallowed warnings and a seven-row page.
    """
    module = task()
    callback = LagAttnRwsPlotCallback(tmp_path, num_examples=1)
    trainer = _trainer_with_batch(stub_batch)
    module.warmup_budget = None  # the run-level figure is a separate seam, tested separately
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
        assert [message for message in warnings if "input rows skipped" in message] == []
        titled = [ax.get_title() for ax in figures[0].axes if ax.get_title()]
        assert len(titled) == _PAGE_ROWS, titled
        for prefix in _INPUT_ROWS:
            assert _axes_titled(figures[0], prefix).has_data()
    finally:
        for figure in figures:
            plt.close(figure)


def test_the_drawn_stream_is_the_encoders_own_input_gate_then_warm_up_mask(task, stub_batch):
    r"""Not ``gate(values)``, which is what the shipped builder draws and is one layer short here.

    This family's gate is a pure gather -- the warm-up is a leading mask, not a shift -- and the
    masking happens inside the availability adapter. Compared against the adapter's **own** buffer
    rather than against a mask rebuilt from $W'$, because a second construction of "the same"
    pattern is exactly how a figure comes to draw a region the model did not mask.
    """
    module = task()
    model = module.orig_model
    inputs = module._build_forward_inputs(stub_batch)

    panels = sample_page.causal_stream_panels(model, inputs, sample_index=0)

    with torch.no_grad():
        gathered = model.target_gate(torch.cat([inputs[0], inputs[1]], dim=-1))
        expected = (gathered * model.target_adapter.availability)[0]
    assert np.allclose(panels[0].values, expected.numpy())
    # And the mask is not vacuous on this fixture: something was actually zeroed.
    assert (expected.numpy() == 0.0).any()


def test_a_value_planted_inside_the_warm_up_reaches_no_pixel_of_the_row(task, stub_batch):
    """The row's whole claim, driven rather than inspected: whatever the shard stored inside
    $[0, W'_c)$ -- real floats on no defined scale, not a zero fill -- is gone by the time the
    encoder, and therefore the figure, sees it."""
    module = task()
    model = module.orig_model
    inputs = list(module._build_forward_inputs(stub_batch))
    warmup = np.asarray(model.target_warmup_steps, dtype=int)
    channel = int(np.argmax(warmup))
    assert warmup[channel] > 0, "a zero-warm-up channel would make this vacuous"
    declared = int(model.target_gate.keep_index[channel])

    planted = torch.cat([inputs[0], inputs[1]], dim=-1)
    planted[:, : warmup[channel], declared] = 1.0e9
    inputs[0], inputs[1] = planted[..., :36], planted[..., 36:]

    panels = sample_page.causal_stream_panels(model, inputs, sample_index=0)

    assert np.all(panels[0].values[: warmup[channel], channel] == 0.0)
    assert np.isfinite(panels[0].values).all()


def test_the_staircase_is_the_warm_up_and_says_so(task, stub_batch):
    r"""``InputStreamPanel.delays`` carries $W'$ rather than a delay, and the existing ``ax.step``
    artist draws it with no new artist of any kind. The label is a panel field for exactly this
    reason: a staircase captioned "first step with data" would name a zero fill that does not
    exist here, on the only part of the row that says what the step function is."""
    module = task()
    model = module.orig_model
    figure = _render(module, stub_batch)
    try:
        ax = _axes_titled(figure, _INPUT_ROWS[0])
        steps = _labelled(ax, sample_page.WARMUP_STAIRCASE_LABEL)

        assert len(steps) == 1
        assert steps[0].get_ydata().tolist() == list(range(model.target_gate.out_channels))
        assert np.allclose(
            steps[0].get_xdata(),
            np.asarray(model.target_warmup_steps, dtype=float) * 4.0,
        )
        # The shipped label is not what is drawn here, and the shipped default is untouched.
        assert sample_page.WARMUP_STAIRCASE_LABEL != shared_page.DELAY_STAIRCASE_LABEL
    finally:
        plt.close(figure)


def test_the_row_title_reports_the_budget_the_counts_and_the_range_in_seconds(task, stub_batch):
    """What a reader needs to size the guard without holding the config beside the page."""
    module = task()
    model = module.orig_model
    panels = sample_page.causal_stream_panels(
        model, module._build_forward_inputs(stub_batch), sample_index=0
    )
    kept = model.target_gate.out_channels

    title = panels[0].title
    assert f"{kept}/{CAUSAL_C_Y} channels kept" in title
    assert f"{CAUSAL_C_Y - kept} dropped" in title
    assert f"{max(model.target_warmup_steps) * 4.0:g} s" in title
    # The source is never gated, and the row says so rather than leaving it to be inferred.
    assert f"{CAUSAL_C_U}/{CAUSAL_C_U} channels kept, 0 dropped" in panels[1].title


def test_the_block_dividers_are_in_surviving_channel_coordinates(task, stub_batch):
    """Under a budget the two stored blocks lose different numbers of channels, so the boundary in
    the gathered stream is not where the declared widths put it -- and a divider drawn at the
    declared split would sit inside the first block."""
    module = task()
    model = module.orig_model
    panels = sample_page.causal_stream_panels(
        model, module._build_forward_inputs(stub_batch), sample_index=0
    )
    keep = np.asarray([int(value) for value in model.target_gate.keep_index])
    expected = int(np.count_nonzero(keep < model.TARGET_BLOCK_SPLIT))

    names = [name for name, _start, _stop in panels[0].blocks]
    starts = [start for _name, start, _stop in panels[0].blocks]
    assert names == ["fhr_st", "fhr_ph"]
    assert starts == [0, expected]
    assert panels[0].blocks[-1][2] == model.target_gate.out_channels


def test_a_source_built_without_its_first_block_is_one_span_over_the_whole_width():
    """``use_up_st: false`` hands the builder the second block's name **alone**.

    That arrangement is refused only when a warm-up budget is also configured, so without one it
    reaches here -- and a builder that indexed the second name unconditionally raised
    ``IndexError`` rather than drawing a row. The page callback swallows it, so the cost was both
    input rows behind one warning and a green suite.
    """
    declared = 15
    spans = sample_page._stream_blocks(
        SOURCE_BLOCKS[1:], 0, np.arange(declared), declared
    )
    assert spans == (("up_ph", 0, declared),)

    # Not vacuous: the two-block form still splits at the boundary, and a block the budget emptied
    # is dropped rather than drawn as a zero-width span.
    both = sample_page._stream_blocks(SOURCE_BLOCKS, 4, np.arange(declared), declared)
    assert both == (("up_st", 0, 4), ("up_ph", 4, declared))
    assert sample_page._stream_blocks(
        SOURCE_BLOCKS, 4, np.arange(4, declared), declared
    ) == (("up_ph", 0, declared - 4),)


def test_it_refuses_a_stream_whose_width_is_not_this_models_own(task, stub_batch):
    """The block spans and the warm-up vector are positional into the declared width, so a stream
    of another width would draw one model's data under another's channel labels."""
    module = task()
    inputs = list(module._build_forward_inputs(stub_batch))
    inputs[2] = inputs[2][..., :-1]

    with pytest.raises(ValueError, match="positional into that width"):
        sample_page.causal_stream_panels(module.orig_model, inputs, sample_index=0)

    with pytest.raises(ValueError, match="at least the"):
        sample_page.causal_stream_panels(module.orig_model, inputs[:2], sample_index=0)


# =================================================================================================
# The forecast rows and the anchor overlay
# =================================================================================================
def test_the_page_rows_all_share_one_time_axis(task, stub_batch):
    """Seven inherited, the two input rows, and the six of ``CAUSAL_EXTRA_ROWS``; the seam replaces
    two of the seven, adds six below them, and must not touch the layout or the axis. A column of
    the page is one instant on every row, which is what lets a reader carry a feature of the
    forecast down into the lag map -- and it is what an added row is most likely to break."""
    figure = _render(task(), stub_batch)
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


def test_each_drawn_window_carries_the_forecast_of_the_anchor_it_is_drawn_at(task, stub_batch):
    r"""The failure the whole seam exists to prevent, and the only one with no exception in it.

    The forecast tensor is $(A_{\max}, H, C)$ indexed by **position in the decoded set**; the step
    a window covers is $[t + 1, t + 1 + H)$ for the *anchor* ``anchor_index`` names at that
    position. The two coincide only at floor $0$ and stride $1$, and the inherited two-sided
    implementation assumes exactly that -- so on this model it would draw a real forecast a floor's
    worth of steps early, at the right shape, in the right colours, on the right axis.
    """
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors = pieces["outs"]["anchor_index"][0].numpy()
        valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
        positions = sample_page._tiling_anchors(anchors, valid, geometry.horizon)
        assert len(positions) > 1, "one window would make the placement assertion vacuous"

        ax = _axes_titled(figure, "Forecast")
        # The full branch's mean line of lane 0: the third line drawn per lane, and the lanes are
        # offset, so the comparison is against the drawn curve minus its own offset.
        full_line = [
            line for line in ax.lines
            if str(line.get_label()).startswith("full ($z^q$")
        ]
        assert len(full_line) == 1
        drawn = np.asarray(full_line[0].get_ydata(), dtype=float)

        # Which channel lane 0 is, recovered from the axis label rather than recomputed.
        declared = int(ax.get_yticklabels()[0].get_text().split("\n")[0].split()[-1])
        keep = [int(value) for value in module.orig_model.target_gate.keep_index]
        channel = keep.index(declared)

        for position in positions:
            anchor = int(anchors[position])
            expected = pieces["outs"]["mu_full"][0, position, :, channel].numpy()
            assert np.allclose(
                drawn[anchor + 1 : anchor + 1 + geometry.horizon], expected, atol=1e-5
            ), position
        # And nothing at all is drawn before the first anchor's own window.
        assert np.all(np.isnan(drawn[: int(anchors[positions[0]]) + 1]))
    finally:
        plt.close(figure)


def test_uncovered_steps_are_gaps_rather_than_a_fabricated_continuation(task, stub_batch):
    """The tiling leaves the warm-up prefix and whatever tail is not a whole window undrawn. Those
    spans are absent, not predicted, and a line drawn through them would read as a forecast the
    model never made."""
    module = task()
    figure = _render(module, stub_batch)
    try:
        ax = _axes_titled(figure, "Forecast")
        floor = module.orig_model.warmup_period
        for line in ax.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            if values.size != module.orig_model.geometry.t:
                continue  # the overlay's own artists, which are not curves
            assert np.all(np.isnan(values[:floor]))
            assert np.isfinite(values).any()
    finally:
        plt.close(figure)


def test_the_overlay_draws_the_decoded_anchors_and_the_training_tiling(task, stub_batch):
    r"""Both, because they are two different sets and the page is produced at only one of them.

    Validation decodes every valid anchor at stride $1$, which is what makes the page reproducible;
    training decodes $\{F + \varphi + kS\}$, roughly a fifteenth as many, at a phase derived per
    segment per epoch. An overlay showing only the dense set says nothing about the geometry the
    gradients were computed at, which is the row's purpose.
    """
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        seconds_per_step = stub_batch.fhr.shape[1] / _FS_RAW / geometry.t
        ax = _axes_titled(figure, "Forecast")

        rug = _labelled(ax, "decoded anchors")
        assert len(rug) == 1
        anchors = pieces["outs"]["anchor_index"][0].numpy()
        valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
        assert np.allclose(rug[0].get_xdata(), anchors[valid] * seconds_per_step)

        tiles = _labelled(ax, "training tiles")
        assert len(tiles) == 1, "one legend entry, whatever the tile count"
        assert f"$S$={TINY_STRIDE}" in str(tiles[0].get_label())
        assert tiles[0].get_xdata()[0] == pytest.approx(
            geometry.warmup * seconds_per_step
        )

        floor = _labelled(ax, "anchor floor")
        assert len(floor) == 1
        assert floor[0].get_xdata()[0] == pytest.approx(geometry.warmup * seconds_per_step)
    finally:
        plt.close(figure)


def test_the_overlay_is_read_from_the_forward_rather_than_recomputed(task, stub_batch):
    """So the figure cannot disagree with the loss. Driven by handing the page an anchor set that
    the geometry alone would never produce: a recomputing implementation draws the geometry's
    anchors and passes, a reading one draws these."""
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
    finally:
        plt.close(figure)


def test_the_error_map_describes_the_anchor_the_row_shades(task, stub_batch):
    r"""The map is one anchor's, and which one has to be findable on the row. Its values are that
    window's absolute error, recomputed here from the target and the forward rather than read back
    -- and gathered at the anchor's own steps, which is the placement assertion again on the panel
    that would be silently wrong in the same way."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors = pieces["outs"]["anchor_index"][0].numpy()
        valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
        positions = sample_page._tiling_anchors(anchors, valid, geometry.horizon)
        position = positions[len(positions) // 2]
        anchor = int(anchors[position])
        seconds_per_step = stub_batch.fhr.shape[1] / _FS_RAW / geometry.t

        ax = _axes_titled(figure, "Forecast")
        spans = [
            (patch.get_x(), patch.get_x() + patch.get_width())
            for patch in ax.patches
            if patch.get_x() > 0.0
        ]
        assert len(spans) == 1, "one shaded window, marking the anchor the error map draws"
        assert spans[0] == pytest.approx(
            (
                (anchor + 1) * seconds_per_step,
                (anchor + 1 + geometry.horizon) * seconds_per_step,
            )
        )

        keep = [int(value) for value in module.orig_model.target_gate.keep_index]
        expected = np.abs(
            pieces["target"][0, anchor + 1 : anchor + 1 + geometry.horizon, keep].numpy()
            - pieces["outs"]["mu_full"][0, position].numpy()
        ).T
        inset = ax.child_axes[0]
        assert np.allclose(inset.images[0].get_array(), expected, atol=1e-5)
    finally:
        plt.close(figure)


# =================================================================================================
# The channel axis, and the field rows drawn on it
# =================================================================================================
#: Every panel on the page whose y-axis is a target or source **channel**, by title prefix. The
#: latent, KL and lag panels are deliberately absent: their axes are latent dimensions and lags.
_CHANNEL_AXES = _INPUT_ROWS + (
    "true $Y^{+}$",
    "base $\\mu^p$",
    "full $\\mu^q$",
    "Source skill",
    "Predicted $\\sigma^q$",
)


def test_every_channel_axis_on_the_page_puts_coefficient_zero_at_the_top(task, stub_batch):
    r"""Channel $0$ at the top and increasing downward, so the scattering block sits above the
    phase block and a reader carries a channel index down the page without it moving. Asserted on
    the extent **and** the origin together: the extent alone flips the axis and leaves the array
    drawn upside down under it, which looks right on a symmetric field and is wrong on every
    other."""
    module = task()
    figure = _render(module, stub_batch)
    try:
        for prefix in _CHANNEL_AXES:
            image = _axes_titled(figure, prefix).images[0]
            _left, _right, bottom, top = image.get_extent()
            channels = image.get_array().shape[0]
            assert image.origin == "upper", prefix
            assert (bottom, top) == pytest.approx((channels - 0.5, -0.5)), prefix
        # And the per-anchor error map, which is an untitled inset of the forecast row.
        inset = _axes_titled(figure, "Forecast").child_axes[0].images[0]
        assert inset.origin == "upper"
        assert inset.get_extent()[2] > inset.get_extent()[3]
    finally:
        plt.close(figure)


def test_the_field_rows_draw_the_same_tiling_the_lane_row_does(task, stub_batch):
    r"""Both branches over **every** kept channel, consecutive and non-overlapping, from the
    tiling the lane row above them already resolved. Three of $98$ channels cannot distinguish a
    model that forecasts a few easy coefficients well from one that is uniformly mediocre, which
    is what these rows exist to show -- but only if they are the same windows."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors = pieces["outs"]["anchor_index"][0].numpy()
        valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
        positions = sample_page._tiling_anchors(anchors, valid, geometry.horizon)
        keep = [int(value) for value in module.orig_model.target_gate.keep_index]

        drawn = {
            name: _axes_titled(figure, prefix).images[0].get_array()
            for name, prefix in (
                ("truth", "true $Y^{+}$"),
                ("base", "base $\\mu^p$"),
                ("full", "full $\\mu^q$"),
            )
        }
        for field in drawn.values():
            assert field.shape == (len(keep), geometry.t)

        for position in positions:
            anchor = int(anchors[position])
            window = slice(anchor + 1, anchor + 1 + geometry.horizon)
            assert np.allclose(
                drawn["truth"][:, window], pieces["target"][0, window, keep].numpy().T, atol=1e-5
            ), position
            for name in ("base", "full"):
                assert np.allclose(
                    drawn[name][:, window],
                    pieces["outs"][f"mu_{name}"][0, position].numpy().T,
                    atol=1e-5,
                ), (name, position)

        # Outside the drawn windows there is no forecast, and every field says so with a gap
        # rather than with a fabricated continuation.
        for field in drawn.values():
            assert np.all(np.ma.getmaskarray(field)[:, : int(anchors[positions[0]]) + 1])
        # One colour scale across the three, so a branch cannot be rescaled into looking right.
        limits = {
            _axes_titled(figure, prefix).images[0].get_clim()
            for prefix in ("true $Y^{+}$", "base $\\mu^p$", "full $\\mu^q$")
        }
        assert len(limits) == 1
    finally:
        plt.close(figure)


def test_the_skill_row_is_the_gap_resolved_per_channel_and_per_step(task, stub_batch):
    """Signed, and on a symmetric diverging scale, because the interesting failure is the region
    where conditioning on the source made the forecast *worse* -- which no scalar on the page can
    show and an unsigned map would hide.

    The full branch is displaced before drawing. At initialisation the posterior *is* the prior --
    the delta head starts at zero -- so the two branches decode one tensor and every skill cell is
    exactly $0$; asserted against that, this test would pass on an implementation that drew zeros
    unconditionally."""
    module = task()
    pieces = _forward(module, stub_batch)
    displaced = pieces["outs"]["mu_full"] + torch.linspace(
        -0.4, 0.4, pieces["outs"]["mu_full"].shape[-1]
    )
    pieces["outs"] = dict(pieces["outs"], mu_full=displaced)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        fields = {
            name: _axes_titled(figure, prefix).images[0].get_array()
            for name, prefix in (
                ("truth", "true $Y^{+}$"),
                ("base", "base $\\mu^p$"),
                ("full", "full $\\mu^q$"),
            )
        }
        image = _axes_titled(figure, "Source skill").images[0]
        expected = np.abs(fields["truth"] - fields["base"]) - np.abs(
            fields["truth"] - fields["full"]
        )
        assert np.allclose(
            np.ma.filled(image.get_array(), 0.0), np.ma.filled(expected, 0.0), atol=1e-5
        )
        assert np.any(np.abs(np.ma.filled(expected, 0.0)) > 1e-3), "the displacement did not land"
        low, high = image.get_clim()
        assert low == pytest.approx(-high)
        # The robust edge, not the maximum: the difference of two absolute errors is heavy-tailed
        # in exactly the way one outlying cell washes the whole map to white.
        finite = np.ma.compressed(image.get_array())
        assert high == pytest.approx(float(np.percentile(np.abs(finite), 99.0)))
    finally:
        plt.close(figure)


def test_the_per_window_score_is_the_objectives_own_number(task, stub_batch):
    r"""Not a second implementation of the loss. The row's two curves go through the objective's
    own :func:`raw_sample_score` under its own :func:`forecast_mask`, so a window's height is in
    the same nats as the ``nll_base_block`` and ``nll_full_block`` in the page's title and their
    difference is ``pred_gap`` restricted to that window. Recomputed here from the forward and the
    batch -- an error curve drawn from a re-derived formula is the one diagnostic that can
    disagree with the run it is diagnosing."""
    module = task()
    pieces = _forward(module, stub_batch)
    figure = _render(module, stub_batch, pieces=pieces)
    try:
        model = module.orig_model
        geometry = model.geometry
        anchors = pieces["outs"]["anchor_index"][0].numpy()
        valid = pieces["outs"]["anchor_valid"][0].numpy().astype(bool)
        positions = sample_page._tiling_anchors(anchors, valid, geometry.horizon)
        keep = torch.as_tensor(
            [int(value) for value in model.target_gate.keep_index], dtype=torch.long
        )

        mask, _coverage = raw_masks.forecast_mask(
            stub_batch.weight,
            geometry,
            coverage_floor=float(model.coverage_floor),
            anchors=pieces["outs"]["anchor_index"],
            anchor_valid=pieces["outs"]["anchor_valid"],
        )
        ax = _axes_titled(figure, "Per-window forecast score")
        for branch, label in (("base", "$D_0$ base"), ("full", "$D_1$ full")):
            drawn = np.asarray(_labelled(ax, label)[0].get_ydata(), dtype=float)
            assert drawn.size == len(positions)
            for lane, position in enumerate(positions):
                anchor = int(anchors[position])
                window = slice(anchor + 1, anchor + 1 + geometry.horizon)
                expected = (
                    losses.raw_sample_score(
                        pieces["outs"][f"mu_{branch}"][0, position],
                        pieces["target"][0, window].index_select(1, keep),
                        likelihood=module.hparams["likelihood"],
                        logvar=pieces["outs"][f"logvar_{branch}"][0, position],
                    )
                    * mask[0, position][:, None]
                ).sum()
                assert drawn[lane] == pytest.approx(float(expected), rel=1e-5), (branch, position)
    finally:
        plt.close(figure)


def test_the_score_row_survives_a_batch_with_no_validity_signal(task, stub_batch):
    """The mask is a function of ``weight``, and scoring an invalid span would put a spike on the
    row the objective never saw. So the row is annotated rather than raised over: it keeps its
    title, its axis and its place, the rows below stay column-aligned, and the gap is visible."""
    import types

    module = task()
    # The forward still needs the weight; what is withheld is the *batch the page is handed*, so
    # only the row under test loses its input.
    pieces = _forward(module, stub_batch)
    stripped = types.SimpleNamespace(
        **{name: value for name, value in vars(stub_batch).items() if name != "weight"}
    )
    figure = _render(module, stub_batch, pieces=pieces, batch=stripped)
    try:
        ax = _axes_titled(figure, "Per-window forecast score")
        assert len(ax.lines) == 0
        assert any("no validity signal" in text.get_text() for text in ax.texts)
        assert len([child for child in figure.axes if child.get_title()]) == _PAGE_ROWS
    finally:
        plt.close(figure)


def test_every_inset_sits_in_the_span_this_tiling_leaves_blank(task):
    r"""Which corner is blank is a property of the tiling, not of the panel. The two-sided page
    stops short of the recording's end and puts its error map in the right margin; this tiling
    starts at the anchor floor $F$ and runs to the end, so the blank span is the *prefix* -- and
    the inherited box put the panel over the last windows of the very forecast it details.

    At the **shipped** geometry, where the claim is the production one: $F = 134$ of $300$ steps,
    so the prefix is a comfortable $44\%$ of the row. The tiny fixture's floor is too small to
    hold a legible inset at all, which is what ``_PREFIX_MIN_SPAN`` is for."""
    from .conftest import SHIPPED_SEQUENCE_LENGTH, shipped_warmup_kwargs

    module = task(model_kwargs=shipped_warmup_kwargs())
    figure = _render(module, make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH))
    try:
        geometry = module.orig_model.geometry
        floor = geometry.warmup / geometry.t
        assert floor > sample_page._PREFIX_MIN_SPAN, "the shipped prefix must hold the insets"
        for title, expected in (
            ("Forecast", 1),                     # the per-anchor error map
            ("Per-window forecast score", 2),    # the error and coverage profiles
        ):
            ax = _axes_titled(figure, title)
            assert len(ax.child_axes) == expected, title
            for inset in ax.child_axes:
                # The inset's own span in the parent's axes fractions, which is what a box is
                # expressed in and what the anchor floor is comparable against.
                left, right = ax.transAxes.inverted().transform(
                    inset.transAxes.transform([[0.0, 0.0], [1.0, 0.0]])
                )[:, 0]
                assert 0.0 <= left < right <= floor, (title, left, right)
        # And the two-sided sibling keeps its own margin: the default is unchanged, so the page
        # that box is right for did not move with this one.
        assert feature_page._ERROR_MAP_BOX[0] > 0.5
    finally:
        plt.close(figure)


def test_the_profiles_carry_one_point_per_kept_channel(task, stub_batch):
    r"""Per-channel error and $2\sigma$ coverage over the same drawn windows, on the page's own
    top-down channel axis, as insets of the *line* row -- an inset over a field row would hide the
    span it covers, and this row carries one marker per drawn window whatever the run does."""
    module = task()
    figure = _render(module, stub_batch)
    try:
        kept = module.orig_model.decoder_out_channels
        insets = _axes_titled(figure, "Per-window forecast score").child_axes
        assert len(insets) == 2, "the error profile and the coverage profile"
        for inset in insets:
            assert inset.get_title() == "", "a titled axes on this page spans the recording"
            # Top-down, like every channel axis above it.
            assert inset.get_ylim() == pytest.approx((kept - 0.5, -0.5))
            branches = [line for line in inset.lines if str(line.get_label()) in ("base", "full")]
            assert len(branches) == 2
            for line in branches:
                assert np.asarray(line.get_ydata()).size == kept
        # The coverage profile alone carries the nominal band, which is derived from the drawn
        # band width rather than written as a number.
        reference = [
            line
            for inset in insets
            for line in inset.lines
            if line.get_xdata()[0] == pytest.approx(sample_page._NOMINAL_COVERAGE)
        ]
        assert len(reference) == 1
    finally:
        plt.close(figure)


def test_every_reserved_extra_row_is_drawn_and_every_drawn_row_is_reserved(task, stub_batch):
    """The two halves live in one constant on purpose. A name reserved and not drawn is a blank
    row on every page of the run; a name drawn and not reserved is a ``KeyError`` raised inside a
    handler that swallows it, i.e. a page silently missing from the whole run."""
    module = task()
    assert module.forecast_extra_rows is sample_page.CAUSAL_EXTRA_ROWS

    figure = _render(module, stub_batch)
    try:
        titled = [ax for ax in figure.axes if ax.get_title()]
        assert len(titled) == _PAGE_ROWS
        for _name, height in sample_page.CAUSAL_EXTRA_ROWS:
            assert height > 0.0
        # Each reserved row drew something; the layout puts them between the forecast row and the
        # input rows, which is what keeps the input rows against the latent they feed.
        order = [ax.get_title() for ax in titled]
        forecast_at = next(i for i, title in enumerate(order) if title.startswith("Forecast"))
        first_input = next(i for i, title in enumerate(order) if title.startswith(_INPUT_ROWS[0]))
        assert first_input - forecast_at == len(sample_page.CAUSAL_EXTRA_ROWS) + 1
    finally:
        plt.close(figure)


def test_the_page_carries_the_physical_delay_caveat(task, stub_batch):
    r"""One-sidedness and zero latency are different properties and this family buys only the
    first. The forecast claim needs no correction -- a coefficient at $t$ is a function of the past
    -- but a peak at lag $\ell$ is an attribution over stored coefficients, and the correction to a
    physical delay is channel-dependent and of the same order as the lag search itself. Asserted as
    a string, so it cannot be dropped by an edit that keeps the figure rendering."""
    figure = _render(task(), stub_batch)
    try:
        drawn = [text.get_text() for text in figure.texts]
        assert sample_page.LAG_TIME_CAVEAT in drawn
        for token in ("stored-coefficient time", "group delay", "$-20$ s"):
            assert token in sample_page.LAG_TIME_CAVEAT
    finally:
        plt.close(figure)


def test_the_channel_rule_and_the_keep_index_mapping_are_the_siblings(task, stub_batch):
    """Reused by object identity rather than reimplemented. The rule is what replaced a
    ``forecast_channels`` config key, which began naming different coefficients the moment a block
    width changed -- and no key like it is reintroduced here."""
    assert sample_page.select_forecast_channels is figure_primitives.select_forecast_channels
    assert sample_page._resolved_keep_index is feature_page._resolved_keep_index
    assert sample_page.FORECAST_CHANNELS is feature_page.FORECAST_CHANNELS

    with pytest.raises(ValueError, match="positional into the declared target stream"):
        sample_page._resolved_keep_index([0, 5], width=3)

    configs = Path(sample_page.__file__).parent / "configs"
    for path in sorted(configs.glob("*.yaml")):
        assert "forecast_channels" not in path.read_text(encoding="utf-8"), path.name


def test_the_lanes_carry_the_truth_and_both_forecasts_with_their_bands(task, stub_batch):
    r"""Three lanes, each with the true coefficient, the base ($z^p$) and full ($z^q$) means and
    both $\pm 2\sigma$ bands, one legend entry per role rather than per lane, plus the overlay's
    three. The counts are what catch a band silently dropped or a lane drawn twice."""
    figure = _render(task(), stub_batch)
    try:
        ax = _axes_titled(figure, "Forecast")
        assert len(ax.collections) == 2 * sample_page.FORECAST_CHANNELS
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "true $Y^{+}$",
            "base ($z^p$, target-only)",
            "full ($z^q$, source-conditioned)",
            "anchor floor $F$=5",
            "decoded anchors (15)",
            "training tiles, $S$=4, $\\varphi$=0",
        ]
        offsets = sorted(ax.get_yticks())
        assert len(set(offsets)) == sample_page.FORECAST_CHANNELS
        assert offsets[0] == pytest.approx(0.0)
    finally:
        plt.close(figure)


def test_it_renders_at_the_shipped_geometry(task):
    """The tiny fixture is a $24$-step window; production is $300$ steps, a $98$-channel decoder
    and $152$ decoded anchors, where the error map is a $98 \\times 15$ image. A page that renders
    only at the test geometry is not a page."""
    from .conftest import SHIPPED_HORIZON, SHIPPED_SEQUENCE_LENGTH, shipped_warmup_kwargs

    module = task(model_kwargs=shipped_warmup_kwargs())
    batch = make_stub_batch(2, SHIPPED_SEQUENCE_LENGTH)
    figure = _render(module, batch)
    try:
        assert module.orig_model.decoder_out_channels == 98
        ax = _axes_titled(figure, "Forecast")
        assert ax.child_axes[0].images[0].get_array().shape == (98, SHIPPED_HORIZON)
        assert len([child for child in figure.axes if child.get_title()]) == _PAGE_ROWS
        # And the field rows carry the same 98 channels over the whole recording, so the page is
        # one channel axis from the input rows down to the error map.
        for row in ("true $Y^{+}$", "base $\\mu^p$", "full $\\mu^q$", "Source skill"):
            image = _axes_titled(figure, row).images[0]
            assert image.get_array().shape == (98, SHIPPED_SEQUENCE_LENGTH), row
        assert _labelled(ax, "decoded anchors")[0].get_xdata().size == (
            SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON - module.orig_model.warmup_period
        )
    finally:
        plt.close(figure)


def test_an_ungated_model_still_draws_both_rows(task, stub_batch):
    """Without a budget there is no gate, no warm-up mask and no availability buffer -- and the
    rows are still the rows. A builder that reached for either unconditionally would fail on the
    arm the whole family is compared against."""
    module = task(model_kwargs=dict(tiny_warmup_kwargs(), **_UNGATED))
    figure = _render(module, stub_batch)
    try:
        for prefix in _INPUT_ROWS:
            assert _axes_titled(figure, prefix).has_data()
        panels = sample_page.causal_stream_panels(
            module.orig_model, module._build_forward_inputs(stub_batch), sample_index=0
        )
        assert panels[0].values.shape[1] == CAUSAL_C_Y
        assert not _labelled(_axes_titled(figure, _INPUT_ROWS[0]), "warm-up")
    finally:
        plt.close(figure)


#: The four keywords that make a model ungated, as a keyword set the guarded tiny kwargs are
#: overridden with. Written out rather than reached for by name, because "no budget" is four
#: absences and a subset of them is a model the constructor refuses.
_UNGATED = dict(
    target_keep_index=None,
    target_warmup_steps=None,
    source_keep_index=None,
    source_warmup_steps=None,
)


# =================================================================================================
# The reduced page
# =================================================================================================
#: The reduced page's rows by title prefix, in the order the layout draws them. Written out here
#: rather than derived from :data:`~teb_vae.lag_attn_cfs.sample_page.COMPACT_PAGE_ROWS`, because a
#: row *name* and the row a reader actually sees are what this file exists to keep equal: deriving
#: the expectation from the same tuple the builder is given would assert only that the tuple equals
#: itself.
_COMPACT_ROW_TITLES = (
    "Raw target FHR",
    "Model input — target",
    "Target-only latent state",
    "$K_t$",
    "Lag attention",
)


def test_the_reduced_page_is_the_five_rows_it_names_in_the_full_page_s_order(task, stub_batch):
    """The reduced page is the full page with rows removed, not a second page: every row on it is a
    row of the other, drawn by the same code, in the same order. That is what lets a reader who
    knows one read the other -- and it is why the rows are *selected* rather than re-listed."""
    figure = _render(module_page_rows(task), stub_batch, rows=sample_page.COMPACT_PAGE_ROWS)
    try:
        titled = [child for child in figure.axes if child.get_title()]
        assert len(titled) == len(sample_page.COMPACT_PAGE_ROWS)
        for ax, prefix in zip(titled, _COMPACT_ROW_TITLES):
            assert ax.get_title().startswith(prefix), (ax.get_title(), prefix)
            assert ax.has_data(), prefix
    finally:
        plt.close(figure)


def test_the_reduced_page_drops_the_source_stream_and_every_forecast_row(task, stub_batch):
    """The rows it does *not* have, named. The forecast rows come through a seam that swallows
    exceptions to protect the fit, so a seam that ignored the row selection would draw into axes it
    does not own -- or raise where nobody sees it -- and the symptom either way is a page, which is
    why the absences are asserted rather than inferred from the count above."""
    figure = _render(module_page_rows(task), stub_batch, rows=sample_page.COMPACT_PAGE_ROWS)
    try:
        titles = [child.get_title() for child in figure.axes if child.get_title()]
        for absent in (_INPUT_ROWS[1], "Forecast", "true $Y^{+}$", "Source skill",
                       "Per-window forecast score", "Per-dimension source-conditioned KL",
                       "$\\widetilde K_{t,\\ell}$"):
            assert not any(title.startswith(absent) for title in titles), absent
    finally:
        plt.close(figure)


def test_the_reduced_page_still_carries_the_physical_delay_caveat(task, stub_batch):
    """The caveat is what says a lag on this transform is not a physical delay, and the reduced
    page is the one that leads with a lag axis. It is drawn by the forecast seam, which returns
    early here -- so a page that kept the lag row and lost the caveat is the exact failure this
    asserts against."""
    figure = _render(module_page_rows(task), stub_batch, rows=sample_page.COMPACT_PAGE_ROWS)
    try:
        assert sample_page.LAG_TIME_CAVEAT in [text.get_text() for text in figure.texts]
    finally:
        plt.close(figure)


def test_the_lag_attention_row_takes_a_log_colour_scale_only_when_asked(task, stub_batch):
    """The log scale is the reduced page's, and the colour normalisation alone: the lag axis is the
    same linear steps on both. The KL-by-lag row is deliberately left linear -- it is nats, read
    against the KL panels -- so the flag is checked to reach one row and not the other."""
    from matplotlib.colors import LogNorm

    module = module_page_rows(task)
    pieces = _forward(module, stub_batch)

    linear = _render(module, stub_batch, pieces=pieces)
    try:
        for prefix in ("Lag attention", "$\\widetilde K_{t,\\ell}$"):
            assert not isinstance(_axes_titled(linear, prefix).images[0].norm, LogNorm), prefix
    finally:
        plt.close(linear)

    logged = _render(module, stub_batch, pieces=pieces, log_lag_attention=True)
    try:
        image = _axes_titled(logged, "Lag attention").images[0]
        assert isinstance(image.norm, LogNorm)
        assert image.norm.vmin > 0.0 and image.norm.vmin < image.norm.vmax
        # The other lag row keeps the linear scale it is read on.
        assert not isinstance(
            _axes_titled(logged, "$\\widetilde K_{t,\\ell}$").images[0].norm, LogNorm
        )
    finally:
        plt.close(logged)


def test_the_log_normaliser_floors_the_range_and_declines_an_empty_panel():
    """Two properties of the normaliser itself, neither reachable through a rendered page. The
    floor is why one near-zero weight cannot stretch the colormap over decades that hold nothing;
    the ``None`` is why a page whose attention is degenerate is still drawn, on a linear scale,
    rather than lost to an exception inside a callback."""
    values = np.full((4, 8), 1e-9, dtype=np.float64)
    values[0, 0] = 1.0
    norm = shared_page.log_attention_norm(values)
    assert norm is not None
    assert norm.vmax == pytest.approx(1.0)
    assert norm.vmin == pytest.approx(shared_page._LOG_ATTENTION_FLOOR)

    assert shared_page.log_attention_norm(np.zeros((4, 8))) is None
    # A panel whose positive mass sits at one value has no range to stretch over.
    assert shared_page.log_attention_norm(np.full((4, 8), 0.125)) is None


def test_a_row_the_page_does_not_reserve_is_refused_by_name(task, stub_batch):
    """A misspelled row must not produce a page that is quietly missing the panel it was rendered
    for -- an absence nothing in the output reports. The message names both what was asked for and
    what the page has, because the legal set depends on which seams the task declared."""
    with pytest.raises(ValueError) as excinfo:
        _render(module_page_rows(task), stub_batch, rows=("raw", "lag_atten"))
    assert "lag_atten" in str(excinfo.value)
    assert "lag_attn" in str(excinfo.value)


def module_page_rows(task) -> Any:
    """A task at the tiny geometry, for the row-selection tests.

    Args:
        task: The package's task fixture.

    Returns:
        The module. Named rather than inlined because five tests build the same one and the
        selection, not the model, is what each of them varies.
    """
    return task(model_kwargs=tiny_warmup_kwargs())


def test_the_shared_page_is_reached_rather_than_copied():
    """No ``lag_attn_cfs/plotting.py`` and no callback of this package's own: the seams exist so a
    sibling supplies rows and inherits the rest, and a second callback class would be a second
    place for the layout, the cuts and the caption to drift."""
    import importlib

    package = Path(sample_page.__file__).parent

    assert not (package / "plotting.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("teb_vae.lag_attn_cfs.plotting")
    # And the context row is the shared implementation, not a copy of it.
    assert feature_page.raw_context_row is shared_page.raw_context_row
