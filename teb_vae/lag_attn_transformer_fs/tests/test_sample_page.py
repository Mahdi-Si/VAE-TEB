r"""The diagnostic page, reached through two levels of inheritance and no module of this package's own.

No figure code is written here. The page is the shared seven-row builder; two of its rows are the
feature-domain sibling's ``feature_forecast_rows``, and this package's only contribution is that the
task -- itself an empty class body -- binds this net's channel facts into them. So the load-bearing
assertion is the one made on the ``functools.partial``: the keep-index and the block boundary read back
off the object the callback will hand to the builder, rather than inferred from a render that happened
to succeed. A render corroborates it, and a render alone would pass on a page drawn against another
model's channels.

Two things are genuinely new and are what the rest of the file is for.

**The channel facts come from a gate this architecture builds at its own construction site.** The
mixin reaches ``self.target_gate`` for the keep-index, and the conv-Transformer base constructs that
gate itself -- so that the page is labelled with the *declared* channel numbers the reach budget kept,
rather than with positions among the survivors, is a fact this pairing owns.

**The lag axes are read after a draw.** Matplotlib defers a secondary axis's limits to draw time, so an
assertion made before one passes against the default $(0, 1)$ whatever the transform is -- which is how
the historical failure this pairing could repeat went unnoticed: two consumers each reached into the
model for the causal input delay under a name of their own guessing, one of those names did not exist,
and the figure's lag axis was silently short by two minutes. The delay accessor exists on both bases,
so what is new here is the *value* pairing rather than the axis arithmetic: one test, not the sibling's
five.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    lag_compensated_seconds,
)
from teb_vae.lag_attn_fs import sample_page  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import _source_delay_steps  # noqa: E402

from .conftest import (  # noqa: E402
    SHIPPED_KWARGS,
    make_patterned_batch,
    shipped_gated_kwargs,
    tiny_gated_kwargs,
)

#: Plausible raw-signal scales, so the raw context row has something to invert. Only the two raw
#: signals carry statistics: the forecast target is the loader's ``normalize_fields`` output used as
#: delivered, and no model in this family adds a second normalisation for it.
_STATS = {"fhr": {"mean": 140.0, "std": 20.0}, "up": {"mean": 30.0, "std": 10.0}}

#: Raw sampling rate, restated rather than imported: the page's time arithmetic is what is under test,
#: and borrowing its own constant would make the assertions circular.
_FS_RAW = 4.0

#: What the shipped reach budget resolves the target stream to.
_SHIPPED_TARGET_CHANNELS = 78

#: The two lag panels' title prefixes, in the order the page lays them out.
_LAG_PANELS = ("Lag attention", r"$\widetilde K_{t,\ell}$")


def _forward(module: Any, data: Any) -> dict:
    """Run the net once and return everything the page builder needs.

    Separate from :func:`_render` because the forward is **stochastic** -- the full branch decodes a
    reparameterised draw -- so a test comparing a drawn artist against the arrays behind it has to
    compare against *this* forward's, not a second one's.

    Args:
        module: A ``SeqVaeLagAttnTrfFsTask``.
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


def _render(module: Any, data: Any, pieces: Any = None, *, draw: bool = False, **overrides) -> Any:
    """Build the whole page for sample 0, through the seam the callback resolves.

    Args:
        module: A ``SeqVaeLagAttnTrfFsTask``.
        data: The batch to draw from.
        pieces: A :func:`_forward` result to draw, or ``None`` to run one.
        draw: Force a canvas draw before returning, so deferred axis limits are real.
        **overrides: Passed to the builder, e.g. ``normalization_stats=None``.

    Returns:
        The matplotlib ``Figure``. The caller closes it.
    """
    pieces = _forward(module, data) if pieces is None else pieces
    kwargs = dict(
        outs=pieces["outs"],
        kld_per_dim=pieces["kld_per_dim"],
        # The builder's parameter is still named for the raw models' target; what this model passes
        # through it is the concatenated feature stream its loss was computed against.
        fhr_raw=pieces["target"],
        geometry=module.orig_model.geometry,
        sample_index=0,
        epoch=3,
        guid="rec-0001",
        beta=0.25,
        scalars={"pred_gap": 0.5},
        up_raw=data.up,
        normalization_stats=_STATS,
        # The callback's own probe, not a constant: this is the value under test.
        delay_steps=_source_delay_steps(module.orig_model),
        # Exactly what `LagAttnRwsPlotCallback._generate_plots` passes.
        forecast_rows=module.forecast_rows,
        batch=data,
    )
    kwargs.update(overrides)
    figure = plotting.build_diagnostic_figure(**kwargs)
    if draw:
        figure.canvas.draw()
    return figure


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


def _lag_axes(figure: Any) -> List[Tuple[str, Any, Any]]:
    """Return ``(title prefix, panel, secondary axis)`` for each of the two lag panels."""
    found = []
    for prefix in _LAG_PANELS:
        matches = [ax for ax in figure.axes if ax.get_title().startswith(prefix)]
        assert len(matches) == 1, f"expected one {prefix!r} panel, found {len(matches)}"
        panel = matches[0]
        assert len(panel.child_axes) == 1, f"{prefix}: {len(panel.child_axes)} secondary axes"
        found.append((prefix, panel, panel.child_axes[0]))
    return found


# =============================================================================
# The seam: what the partial carries
# =============================================================================
def test_the_partial_binds_this_nets_keep_index_and_block_split(task, shipped_gated):
    """The load-bearing assertion of this file, and it is made on the object rather than on a picture.

    The two bound values are what the page cannot recover from the arrays it is handed. The keep-index
    says which *declared* channel each of the decoder's outputs is, which is needed both to gather the
    truth a lane is judged against and to put a channel number on the axis that still means the same
    thing at another reach budget. The block split is where the two stored blocks meet on that channel
    axis -- the same boundary ``pred_gap_st`` and ``pred_gap_ph`` are reported either side of.

    The gate both are read off is built at *this* architecture's own construction site, which is why
    the pairing is asserted here rather than inherited from either sibling's copy.
    """
    rows = task(model_kwargs=shipped_gated).forecast_rows

    assert rows.func is sample_page.feature_forecast_rows
    assert set(rows.keywords) == {"keep_index", "block_split"}
    assert list(rows.keywords["keep_index"]) == list(shipped_gated["target_keep_index"])
    assert len(rows.keywords["keep_index"]) == _SHIPPED_TARGET_CHANNELS
    assert rows.keywords["block_split"] == 43


def test_the_ungated_arm_binds_no_index_because_there_is_no_gate(task):
    """The decoder then emits every declared channel in order, and there is no gate to read an index
    off at all -- so the page must be handed ``None`` rather than a range it would have to invent."""
    rows = task().forecast_rows

    assert rows.keywords["keep_index"] is None


def test_this_package_ships_no_figure_module():
    """Near-vacuous on the day it is written, and the thing that fails when someone later reaches for a
    local copy. The seam exists so a model supplies two rows and inherits the other five; a second
    ``sample_page`` or ``plotting`` module here would be a second place for the layout, the cuts, the
    caption and the channel rule to drift."""
    import importlib
    from pathlib import Path

    package = Path(__file__).resolve().parents[1]

    assert not (package / "plotting.py").exists()
    assert not (package / "sample_page.py").exists()
    for name in ("plotting", "sample_page"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"teb_vae.lag_attn_transformer_fs.{name}")


# =============================================================================
# The page, rendered
# =============================================================================
def test_the_page_has_seven_rows_on_one_time_axis(task, patterned_batch):
    """The seam replaces two rows; it must not touch the other five, the layout or the axis. A column
    of the page is one instant on every row, which is what lets a reader carry a feature of the
    forecast down into the lag map."""
    figure = _render(task(), patterned_batch)
    try:
        titled = [ax for ax in figure.axes if ax.get_title()]
        assert len(titled) == 7, [ax.get_title()[:30] for ax in titled]
        t_max = patterned_batch.fhr.shape[1] / _FS_RAW
        for ax in titled:
            assert ax.get_xlim() == pytest.approx((0.0, t_max)), ax.get_title()
            assert ax.has_data(), ax.get_title()
        # The five inherited rows, drawn by the builder itself.
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


def test_the_forecast_row_carries_three_data_chosen_channels_named_by_declared_index(task):
    """Two properties in one, because either alone would pass on the wrong figure: the lanes are the
    channel rule's answer for *this* batch, and each is labelled with the channel's position in the
    declared $c_y$ stream rather than its position among the survivors. A guarded run's third surviving
    channel is declared channel $9$ here, and a page labelling it $2$ would name a different
    coefficient at every reach budget."""
    module = task(model_kwargs=tiny_gated_kwargs())
    batch = make_patterned_batch()
    figure = _render(module, batch)
    try:
        ax = _axes_titled(figure, "Forecast")
        labels = [text.get_text().split("\n")[0] for text in ax.get_yticklabels()]
        keep = [int(value) for value in module.orig_model.target_gate.keep_index]

        assert len(labels) == sample_page.FORECAST_CHANNELS == 3
        # The lane *order* is the rule's, so the labels are a permutation of the survivors rather
        # than the survivors in index order.
        assert sorted(labels) == sorted(f"ch {index}" for index in keep)
        assert f"3 of {len(keep)} target channels" in ax.get_title()
    finally:
        plt.close(figure)


def test_the_error_map_is_a_c_keep_by_h_inset_that_never_claims_the_shared_time_axis(
    task, patterned_batch
):
    r"""Its x-axis is the horizon step of one anchor, not physical time. Drawn as a panel of its own it
    would make the forecast curves narrower than the other six rows and break the page's
    one-instant-per-column property; drawn as a titled axes it would break the assertion above. So it
    is an untitled inset, and the row it sits in still spans the recording."""
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
        # 'none', because a resampled map invents values between two channels, and per-channel
        # resolution is the entire reason the panel exists.
        assert image.get_interpolation() == "none"
        # Its colorbar took the row's reserved column, which the raw page leaves hidden.
        assert any("Y^{+}" in ax.get_ylabel() for ax in figure.axes)
    finally:
        plt.close(figure)


def test_row_one_draws_the_raw_trace_from_the_batch_not_the_feature_target(task, patterned_batch):
    """The sharpest confusion this page can make. The inherited row plots the scored target against the
    raw time axis; here that target is a $(B, T, c_y)$ feature block, so the row has to reach into the
    batch instead -- and the two are not even the same length, which is what makes this checkable
    rather than merely plausible."""
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
    finally:
        plt.close(figure)


def test_without_statistics_the_context_row_says_normalised_instead_of_lying(
    task, patterned_batch
):
    """The criterion the page has to meet when the run's loader statistics cannot be reached at all --
    ``normalization_stats_of`` returns ``None``, which is what a run whose stats file moved produces."""
    figure = _render(task(), patterned_batch, normalization_stats=None)
    try:
        ax = _axes_titled(figure, "Raw target FHR")

        assert "normalised" in ax.get_ylabel()
        assert "bpm" not in ax.get_ylabel()
        # The forecast row is *always* in normalised units: the target is the loader's output used as
        # delivered, and there is no second normalisation to invert.
        assert "normalised" in _axes_titled(figure, "Forecast").get_ylabel()
    finally:
        plt.close(figure)


def test_it_renders_at_the_shipped_geometry_and_the_budgets_channel_count(task):
    """The tiny fixture is $109$ channels ungated and $3$ gated; production is $78$ survivors over $300$
    steps, where the error map is a $78 \\times 30$ image and the lanes are picked from a real reach
    budget's keep-index. Not marked slow: it builds and forwards the production net, which is seconds,
    and a page that renders only at the test geometry is not a page."""
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
        # The boundary between the two stored blocks, drawn on the channel axis because the two have
        # different filter reaches and the run reports its forecast gap either side of it.
        keep = np.asarray([int(value) for value in kwargs["target_keep_index"]])
        expected = int(np.count_nonzero(keep < module.orig_model.TARGET_BLOCK_SPLIT))
        boundaries = [line.get_ydata()[0] for line in _error_map(figure).lines]
        assert boundaries == pytest.approx([expected - 0.5])
    finally:
        plt.close(figure)


# =============================================================================
# The lag axes, read after a draw
# =============================================================================
def test_both_lag_panels_carry_the_axis_this_models_reported_delay_implies(task, shipped_gated):
    r"""The pairing this file exists for, and the one thing about the lag panels that is new here.

    Each panel's primary axis is the lag index $\ell$ and its secondary is $4(\ell + \delta)$ seconds.
    Three things must hold at once: the map must be the one the *model's* own $\delta$ gives -- not a
    zero-offset one, which is what an unresolved delay silently produces -- the two panels must carry
    the same map, since they are read together, and the label must say *compensated* rather than a bare
    "Lag (s)", which is ambiguous between the mechanically compensated lag and the uncorrected sensor
    one.

    Read **after a draw**: matplotlib defers a secondary axis's limits to draw time, so an assertion
    made before one passes against the default $(0, 1)$ whatever the transform is -- which would make
    this test pass on exactly the bug it is here to catch.
    """
    module = task(model_kwargs=shipped_gated)
    batch = make_patterned_batch(2, int(SHIPPED_KWARGS["sequence_length"]))
    figure = _render(module, batch, draw=True)
    try:
        delay = int(module.orig_model.source_delay_steps)
        assert delay == _source_delay_steps(module.orig_model) == 30, (
            "an unguarded model makes every equality below trivial"
        )

        seen = []
        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            expected = (
                float(lag_compensated_seconds(low, delay_steps=delay)),
                float(lag_compensated_seconds(high, delay_steps=delay)),
            )
            assert secondary.get_ylim() == pytest.approx(expected), prefix
            assert secondary.get_ylabel() == COMPENSATED_LAG_AXIS_LABEL, prefix
            seen.append(secondary.get_ylim())

        assert seen[0] == pytest.approx(seen[1])
    finally:
        plt.close(figure)


def test_an_unguarded_model_draws_a_zero_offset_axis_and_that_is_the_honest_one(
    task, patterned_batch
):
    """The other direction, and the one where agreement is easy -- which is why the guarded case above
    is what has to be asserted. Without a reach budget there is no delay to compensate, the axis is
    $4\\ell$, and a regression that made the guarded case read zero again would otherwise hide behind
    this passing."""
    module = task()
    figure = _render(module, patterned_batch, draw=True)
    try:
        assert int(module.orig_model.source_delay_steps) == 0

        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            assert secondary.get_ylim() == pytest.approx(
                (
                    float(lag_compensated_seconds(low, delay_steps=0)),
                    float(lag_compensated_seconds(high, delay_steps=0)),
                )
            ), prefix
    finally:
        plt.close(figure)
