r"""The two rows this model replaces, and the rule that decides which channels they draw.

Two kinds of test. The channel rule is driven directly, on hand-made coverage, because what it has
to be is *a rule* -- deterministic, tie-broken, and meaning the same thing at any channel count --
and that is invisible in a rendered figure. The rows are driven through the real page builder with
the real seam the callback resolves, because what they have to be is two rows of a seven-row page
whose other five are the sibling's: a replacement that quietly redrew the layout, rescaled the
shared time axis or lost the raw traces would still produce a plausible-looking figure.

The one assertion that fails loudest if the target domain is mixed up is that row $1$ draws the
**raw** trace: it is the only row on the page this model does not read, its length is the raw grid
rather than the decimated one, and drawing ``rows.target`` there -- which is what the inherited row
does -- is exactly the mistake the seam exists to prevent.
"""
from __future__ import annotations

from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.figure_primitives import select_forecast_channels  # noqa: E402
from teb_vae.lag_attn_fs import sample_page  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws import sample_page as shared_page  # noqa: E402

from .conftest import (  # noqa: E402
    SHIPPED_KWARGS,
    make_patterned_batch,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)

#: Plausible raw-signal scales, so the raw context row has something to invert. Only the two raw
#: signals carry statistics: the forecast target is the loader's ``normalize_fields`` output used
#: as delivered, and this package deliberately adds no second normalisation for it.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: Raw sampling rate, restated rather than imported: the page's second arithmetic is what is under
#: test, and borrowing its own constant would make the assertions circular.
_FS_RAW = 4.0

#: What the shipped reach budget resolves the target stream to. Pinned so the shipped-width render
#: below cannot silently become the unguarded one.
_SHIPPED_TARGET_CHANNELS = 78


def _forward(module: Any, data: Any) -> dict:
    """Run the net once and return everything the page builder needs.

    Separate from :func:`_render` because the forward is **stochastic** -- the full branch decodes
    a reparameterised draw -- so a test comparing a drawn artist against the arrays behind it has
    to compare against *this* forward's, not a second one's.

    Args:
        module: A ``SeqVaeLagAttnFsTask``.
        data: The batch to run on.

    Returns:
        ``{'outs', 'kld_per_dim', 'target'}``.
    """
    model = module.orig_model
    with torch.no_grad():
        outs = model(*module._build_forward_inputs(data))
        target, _weight = module._build_raw_target(data)
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    return {"outs": outs, "kld_per_dim": kld_per_dim, "target": target}


def _render(module: Any, data: Any, pieces: Any = None, **overrides) -> Any:
    """Build the whole page for sample 0, through the seam the callback resolves.

    Args:
        module: A ``SeqVaeLagAttnFsTask``.
        data: The batch to draw from. Named so a test can still override the builder's own
            ``batch`` argument -- which is what the page reads its raw traces through, and is
            therefore a thing worth being able to withhold.
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
        guid="rec-0001",
        beta=0.25,
        scalars={"pred_gap": 0.5},
        up_raw=data.up,
        normalization_stats=_STATS,
        # Exactly what `LagAttnRwsPlotCallback._generate_plots` passes.
        forecast_rows=module.forecast_rows,
        batch=data,
    )
    kwargs.update(overrides)
    return plotting.build_diagnostic_figure(**kwargs)


def _axes_titled(figure: Any, prefix: str) -> Any:
    """Return the single axes whose title starts with ``prefix``."""
    matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} panel, found {len(matches)}"
    return matches[0]


def _error_map(figure: Any) -> Any:
    """Return the forecast row's inset error map."""
    insets = _axes_titled(figure, "Forecast").child_axes
    assert len(insets) == 1, f"expected one inset on the forecast row, found {len(insets)}"
    return insets[0]


def _coverage_block(coverages: List[float], anchors: int = 40):
    """A truth/mean/sigma triple whose per-channel $2\\sigma$ coverage is exactly ``coverages``.

    Built by placing, in each channel, a known fraction of its elements inside the band and the
    rest far outside it -- so the statistic the rule ranks on is dictated rather than measured,
    and a rule that ranked on something else (the mean error, the variance) fails here.

    Args:
        coverages: Desired coverage per channel; each must be a multiple of ``1 / anchors``.
        anchors: Elements per channel.

    Returns:
        ``(truth, mu, sigma)``, each ``(anchors, len(coverages))``.
    """
    mu = np.zeros((anchors, len(coverages)), dtype=np.float64)
    sigma = np.ones_like(mu)
    truth = np.zeros_like(mu)
    for channel, coverage in enumerate(coverages):
        inside = int(round(coverage * anchors))
        truth[inside:, channel] = 100.0  # far outside mu +/- 2 sigma
    return truth, mu, sigma


# =============================================================================
# The channel-selection rule
# =============================================================================
def test_it_returns_the_worst_the_middle_and_the_best_calibrated_channels():
    """Not the worst three. A panel showing only failures reads as a broken model on every run,
    and one showing only successes reads as a working one; the extremes plus the middle is what
    makes the same three lanes informative in both cases."""
    truth, mu, sigma = _coverage_block([0.5, 0.9, 0.1, 0.7, 0.3])

    chosen, coverage = select_forecast_channels(truth, mu, sigma, count=3)

    assert list(chosen) == [2, 0, 1]
    assert coverage == pytest.approx([0.5, 0.9, 0.1, 0.7, 0.3])


def test_it_is_deterministic_and_breaks_ties_by_channel_index():
    """Two runs of the same figure must draw the same lanes, and equal coverage is the common
    case early in training -- every channel at $0$ or at $1$ -- so an unstable tie-break would
    make the lanes hop between epochs for no reason a reader could see."""
    truth, mu, sigma = _coverage_block([0.4, 0.4, 0.4, 0.4])

    first, _ = select_forecast_channels(truth, mu, sigma, count=3)
    second, _ = select_forecast_channels(truth, mu, sigma, count=3)

    # Channel order, at the three evenly spaced positions of a four-long ranking.
    assert list(first) == list(second) == [0, 2, 3]


def test_the_rule_survives_a_change_in_the_channel_count():
    """The failure that removed ``lag_attn``'s ``forecast_channels`` key: the stored
    phase-harmonic block went from $44$ to $66$ channels and every configured index silently began
    naming a different coefficient. A rule cannot have that failure -- appending channels to the
    right of the worst one must leave it the worst one."""
    coverages = [0.5, 0.9, 0.1, 0.7]
    narrow, mu_narrow, sigma_narrow = _coverage_block(coverages)
    wide, mu_wide, sigma_wide = _coverage_block(coverages + [0.8, 0.6, 0.95])

    worst_narrow, _ = select_forecast_channels(narrow, mu_narrow, sigma_narrow, count=3)
    worst_wide, wide_coverage = select_forecast_channels(wide, mu_wide, sigma_wide, count=3)

    assert worst_narrow[0] == worst_wide[0] == 2
    assert len(worst_wide) == 3
    # And the best moved, because a better-calibrated channel now exists -- the rule reads the
    # data rather than the width.
    assert worst_wide[-1] == int(np.argmax(wide_coverage)) == 6


def test_uncovered_positions_are_ignored_and_an_unscorable_channel_sorts_worst():
    r"""The tiled forecast is ``NaN`` wherever no window covers the step, and counting those as
    misses would rank a channel by how much of the recording the tiling reached. A channel with no
    finite element at all is the one worth looking at, so it scores $0$ rather than $1$."""
    truth, mu, sigma = _coverage_block([0.9, 0.5, 0.9], anchors=40)
    mu[:4] = np.nan  # the uncovered prefix the tiling leaves, in every channel
    mu[:, 2] = np.nan  # and one channel that was never scored at all

    chosen, coverage = select_forecast_channels(truth, mu, sigma, count=3)

    assert coverage[2] == 0.0
    assert chosen[0] == 2
    # Read off the 36 covered anchors alone, not off all 40: channel 0's first 36 elements are
    # inside the band, of which 32 survive the cut.
    assert coverage[0] == pytest.approx(32 / 36)
    assert coverage[1] == pytest.approx(16 / 36)


def test_it_refuses_a_block_with_no_channels_and_a_non_positive_count():
    truth, mu, sigma = _coverage_block([0.5])

    with pytest.raises(ValueError, match="channel axis"):
        select_forecast_channels(np.zeros((4, 0)), np.zeros((4, 0)), np.ones((4, 0)))
    with pytest.raises(ValueError, match="count must be"):
        select_forecast_channels(truth, mu, sigma, count=0)


# =============================================================================
# The page
# =============================================================================
def test_the_page_still_has_seven_rows_on_one_time_axis(task, patterned_batch):
    """The seam replaces two rows; it must not touch the other five, the layout or the axis. A
    column of the page is one instant on every row, which is what lets a reader carry a feature of
    the forecast down into the lag map."""
    figure = _render(task(), patterned_batch)
    try:
        titled = [ax for ax in figure.axes if ax.get_title()]
        assert len(titled) == 7, [ax.get_title()[:30] for ax in titled]
        t_max = patterned_batch.fhr.shape[1] / _FS_RAW
        for ax in titled:
            assert ax.get_xlim() == pytest.approx((0.0, t_max)), ax.get_title()
            assert ax.has_data(), ax.get_title()
        # The five inherited rows, unchanged and drawn by the builder itself.
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


def test_row_one_draws_the_raw_trace_from_the_batch_not_the_feature_target(task, patterned_batch):
    """The sharpest confusion this page can make. The inherited row plots ``rows.target`` against
    the raw time axis; here that target is a $(B, T, c_y)$ feature block, so the row has to reach
    into the batch instead -- and the two are not even the same length, which is what makes this
    checkable rather than merely plausible."""
    figure = _render(task(), patterned_batch)
    try:
        ax = _axes_titled(figure, "Raw target FHR")
        drawn = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        expected = (
            patterned_batch.fhr[0].numpy() * (_STATS["fhr"]["std"] + 1e-8) + _STATS["fhr"]["mean"]
        )

        assert drawn.size == patterned_batch.fhr.shape[1]
        assert np.allclose(drawn, expected, atol=1e-3)
        assert "bpm" in ax.get_ylabel()
        # The source trace beside it, which rows 6 and 7 are statements about.
        twins = [child for child in figure.axes if child.get_ylabel().startswith("UP")]
        assert len(twins) == 1 and twins[0].get_ylabel() == "UP (mmHg)"
    finally:
        plt.close(figure)


def test_without_statistics_the_context_row_says_normalised_instead_of_lying(
    task, patterned_batch
):
    """The other direction of the conversion, and the criterion the page has to meet when the
    run's loader statistics cannot be reached at all."""
    figure = _render(task(), patterned_batch, normalization_stats=None)
    try:
        ax = _axes_titled(figure, "Raw target FHR")

        assert "normalised" in ax.get_ylabel()
        assert "bpm" not in ax.get_ylabel()
        # The forecast row is *always* in normalised units: the target is the loader's output used
        # as delivered, and there is no second normalisation to invert.
        assert "normalised" in _axes_titled(figure, "Forecast").get_ylabel()
    finally:
        plt.close(figure)


def test_a_batch_without_the_raw_trace_still_draws_the_other_six_rows(task, patterned_batch):
    """The builder's ``batch`` defaults to ``None`` and the raw traces are context, not a readout.
    Losing them must cost the page one row's content and nothing else -- including its place in
    the layout, so the rows below stay column-aligned."""
    figure = _render(task(), patterned_batch, batch=None, up_raw=None)
    try:
        ax = _axes_titled(figure, "Raw target FHR")

        assert len(ax.lines) == 0
        assert [text.get_text() for text in ax.texts] == ["raw traces unavailable in this batch"]
        assert len([child for child in figure.axes if child.get_title()]) == 7
        assert _axes_titled(figure, "Forecast").has_data()
    finally:
        plt.close(figure)


def test_the_forecast_row_draws_the_channels_the_rule_picks_named_by_declared_index(task):
    """Two properties in one, because either alone would pass on the wrong figure: the lanes are
    the rule's answer for *this* batch, and each is labelled with the channel's position in the
    declared $c_y$ stream rather than its position among the survivors. A guarded run's third
    surviving channel is declared channel $9$ here, and a page labelling it $2$ would name a
    different coefficient at every reach budget."""
    module = task(model_kwargs=tiny_gated_kwargs())
    batch = make_patterned_batch()
    figure = _render(module, batch)
    try:
        ax = _axes_titled(figure, "Forecast")
        labels = [text.get_text() for text in ax.get_yticklabels()]
        keep = [int(value) for value in module.orig_model.target_gate.keep_index]

        assert len(labels) == 3
        assert [label.split("\n")[0] for label in labels] == [f"ch {index}" for index in keep] or (
            # The lane *order* is the rule's, so the labels are a permutation of the survivors
            # rather than the survivors in index order.
            sorted(label.split("\n")[0] for label in labels)
            == sorted(f"ch {index}" for index in keep)
        )
        assert f"3 of {len(keep)} target channels" in ax.get_title()
    finally:
        plt.close(figure)


def test_the_error_map_is_an_inset_that_never_claims_the_shared_time_axis(task, patterned_batch):
    r"""Its x-axis is the horizon step of one anchor, not physical time. Drawn as a panel of its
    own it would make the forecast curves narrower than the other six rows and break the page's
    one-instant-per-column property; drawn as a titled axes it would break the assertion that
    checks it. So it is an untitled inset, and the row it sits in still spans the recording."""
    module = task()
    figure = _render(module, patterned_batch)
    try:
        geometry = module.orig_model.geometry
        inset = _error_map(figure)
        image = inset.images[0]

        assert inset.get_title() == ""
        assert image.get_array().shape == (
            module.orig_model.decoder_out_channels,
            geometry.horizon,
        )
        # The channel axis runs **top-down**: channel 0 at the top, which is the one convention
        # every channel axis in the family reads by, so the map and the input rows two rows above
        # it put the same channel at the same height. Paired with ``origin='upper'``; the extent
        # alone would flip the axis and leave the array drawn upside down under it.
        assert image.get_extent() == pytest.approx(
            tuple(
                shared_page.top_down_extent(
                    -0.5, geometry.horizon - 0.5, module.orig_model.decoder_out_channels
                )
            )
        )
        assert image.origin == "upper"
        # 'none', because a resampled map invents values between two channels, and per-channel
        # resolution is the entire reason the panel exists.
        assert image.get_interpolation() == "none"
        # Its colorbar took the row's reserved column, which the raw page leaves hidden.
        assert any("Y^{+}" in ax.get_ylabel() for ax in figure.axes)
    finally:
        plt.close(figure)


def test_the_error_map_describes_the_anchor_the_row_shades(task, patterned_batch):
    r"""The map is one anchor's, and which one has to be findable on the row: otherwise it is a
    picture of a window the reader cannot locate. The shaded span is the anchor's own forecast
    window, $[t+1, t+1+H)$ on the decimated grid, and the values are that window's absolute
    error -- recomputed here from the target and the forward pass rather than read back."""
    module = task()
    # One forward, drawn and then compared against: the full branch decodes a reparameterised
    # draw, so a second forward would disagree with the figure for a reason that is not a bug.
    pieces = _forward(module, patterned_batch)
    figure = _render(module, patterned_batch, pieces=pieces)
    try:
        geometry = module.orig_model.geometry
        anchors = list(range(geometry.warmup, geometry.t_valid, geometry.horizon))
        anchor = anchors[len(anchors) // 2]
        seconds_per_step = patterned_batch.fhr.shape[1] / _FS_RAW / geometry.t

        ax = _axes_titled(figure, "Forecast")
        # The other span on this row is the warm-up shading the layout hook adds, which starts at
        # zero; this one starts at the anchor's own forecast window.
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

        expected = np.abs(
            pieces["target"][0, anchor + 1 : anchor + 1 + geometry.horizon].numpy()
            - pieces["outs"]["mu_full"][0, anchor].numpy()
        ).T
        assert np.allclose(_error_map(figure).images[0].get_array(), expected, atol=1e-5)
    finally:
        plt.close(figure)


def test_the_lanes_carry_the_truth_and_both_forecasts_with_their_bands(task, patterned_batch):
    r"""Three lanes, each with the true coefficient, the base ($z^p$) and full ($z^q$) means and
    both $\pm 2\sigma$ bands, and one legend entry per role rather than per lane. The counts are
    what catch a band silently dropped or a lane drawn twice."""
    figure = _render(task(), patterned_batch)
    try:
        ax = _axes_titled(figure, "Forecast")

        assert len(ax.lines) == 3 * sample_page.FORECAST_CHANNELS
        assert len(ax.collections) == 2 * sample_page.FORECAST_CHANNELS
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "true $Y^{+}$",
            "base ($z^p$, target-only)",
            "full ($z^q$, source-conditioned)",
        ]
        # The lanes are offset, not overlaid: a shared baseline would make three channels of a
        # normalised stream unreadable as three signals.
        offsets = sorted(ax.get_yticks())
        assert len(set(offsets)) == sample_page.FORECAST_CHANNELS
        assert offsets[0] == pytest.approx(0.0)
    finally:
        plt.close(figure)


def test_it_refuses_a_keep_index_that_is_not_the_decoders_width():
    """A shorter or longer index gathers some other channel's truth into every lane and draws a
    figure that looks exactly right, so it has to fail rather than degrade."""
    with pytest.raises(ValueError, match="positional into the declared target stream"):
        sample_page._resolved_keep_index([0, 5], width=3)

    assert list(sample_page._resolved_keep_index(None, width=4)) == [0, 1, 2, 3]
    assert list(sample_page._resolved_keep_index(torch.tensor([1, 7]), width=2)) == [1, 7]


def test_it_renders_at_the_shipped_geometry_and_the_budgets_channel_count(task):
    """The tiny fixture is $109$ channels ungated and $3$ gated; production is $78$ survivors over
    $300$ steps, where the error map is a $78 \\times 30$ image and the lanes are picked from a
    real reach budget's keep-index. Not marked slow: it builds and forwards the production net,
    which is seconds, and a page that renders only at the test geometry is not a page."""
    kwargs = shipped_gated_kwargs()
    module = task(model_kwargs=kwargs)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    figure = _render(module, batch)
    try:
        assert module.orig_model.decoder_out_channels == _SHIPPED_TARGET_CHANNELS
        assert _error_map(figure).images[0].get_array().shape == (
            _SHIPPED_TARGET_CHANNELS,
            int(SHIPPED_KWARGS["horizon"]),
        )
        assert len([ax for ax in figure.axes if ax.get_title()]) == 7
        # The boundary between the two stored blocks, drawn on the channel axis because the two
        # have different filter reaches and the run reports its forecast gap either side of it.
        keep = np.asarray([int(value) for value in kwargs["target_keep_index"]])
        expected = int(np.count_nonzero(keep < module.orig_model.TARGET_BLOCK_SPLIT))
        boundaries = [line.get_ydata()[0] for line in _error_map(figure).lines]
        assert boundaries == pytest.approx([expected - 0.5])
    finally:
        plt.close(figure)


def test_the_shared_page_is_reached_rather_than_copied():
    """No ``lag_attn_fs/plotting.py`` and no callback of this package's own: the seam exists so a
    sibling supplies two rows and inherits the other five, and a second callback class would be a
    second place for the layout, the cuts and the caption to drift."""
    import importlib
    from pathlib import Path

    package = Path(sample_page.__file__).parent

    assert not (package / "plotting.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("teb_vae.lag_attn_fs.plotting")
    # And row 1 is the sibling's implementation, not a copy of it.
    assert sample_page.raw_context_row is shared_page.raw_context_row
