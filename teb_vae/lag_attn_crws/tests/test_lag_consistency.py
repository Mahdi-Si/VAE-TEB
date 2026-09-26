r"""The figure's lag axis and the model's own reported lag must be the same number.

The raw-signal sibling records the failure this file exists for: two consumers each reached into the
model for the causal input delay $\delta$ under a name of its own guessing, one of those names did
not exist and silently read zero, and the figure's lag axis came out short against the evaluation's.
Both went on producing plausible numbers.

**This cell re-creates that bug class twice over.** The input rows fill
``InputStreamPanel.delays`` with $W'_c$, a **warm-up**, through an attribute whose name says delay;
a warm-up is a leading region a channel is not honest in and shifts nothing, whereas a delay means
the source memory the attention queries is itself $\delta$ steps stale. On an unaligned model the
gate is a pure gather, $\delta = 0$ and the honest axis is $\Delta\ell$ -- the arm in which reading
the panel's staircase as a delay is visible at all. Under an alignment the gate carries the
per-channel shifts $d_c$, $\delta = \max_c d_c > 0$, and the honest axis is $\Delta(\ell + \delta)$
on both lag panels, with the page's delay probe agreeing with the model's own.

``lag_floor`` is the second offset on the same axis and is not a shift either: it restricts
**which** lags a step may read, $\mathbb 1[t - \ell \ge F_u]$, not what a lag index means, so the
axis must not move by the floor while the mask must. Nor may this cell's raw-grid forecast rows move
the lag panels' limits.

The secondary axis is read **after a draw**. Matplotlib defers a secondary axis's limits to draw
time, so an assertion made before one passes against the default $(0, 1)$ whatever the transform is
-- which would make this file pass on exactly the bug it is here to catch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    SECONDS_PER_STEP,
    lag_compensated_seconds,
)
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import _source_delay_steps  # noqa: E402

from .conftest import (  # noqa: E402
    TINY_STRIDE,
    make_stub_batch,
    make_task,
    tiny_align_kwargs,
    tiny_warmup_kwargs,
)

#: The two lag panels' title prefixes, in the order the page lays them out.
_LAG_PANELS = ("Lag attention", r"$\widetilde K_{t,\ell}$")

#: A floor inside the tiny geometry's trained range, so the rows it silences are real rows.
_LAG_FLOOR = 3


def _module_and_batch(aligned: bool = False, **overrides: Any) -> Tuple[Any, Any]:
    """Build this model wrapped in its task, at the tiny warm-up guard and the tiling stride.

    Args:
        aligned: Build at the tiny alignment as well, so the source gate carries a shift.
        **overrides: Constructor keywords applied on top of the guarded tiny set.

    Returns:
        ``(task, batch)``.
    """
    guard = tiny_align_kwargs if aligned else tiny_warmup_kwargs
    module = make_task(model_kwargs=guard(anchor_stride=TINY_STRIDE, **overrides))
    return module, make_stub_batch()


def _page(module: Any, batch: Any, **overrides: Any) -> Any:
    """Draw the whole page through the callback's own seams and force a draw.

    Args:
        module: The task whose net is drawn.
        batch: The batch to draw from.
        **overrides: Passed to the page builder, e.g. a replacement ``forecast_rows``.

    Returns:
        A drawn ``Figure``. The caller closes it.
    """
    model = module.orig_model
    with torch.no_grad():
        inputs = module._build_forward_inputs(batch)
        outs = model(*inputs)
        target, _weight = module._build_raw_target(batch)
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    kwargs = dict(
        outs=outs,
        kld_per_dim=kld_per_dim,
        fhr_raw=target,
        geometry=model.geometry,
        sample_index=0,
        epoch=0,
        guid="SEG000",
        beta=1.0,
        scalars={},
        up_raw=batch.up,
        normalization_stats=None,
        # The callback's own probe, not a constant: this is the value under test.
        delay_steps=_source_delay_steps(model),
        forecast_rows=module.forecast_rows,
        batch=batch,
        input_streams=plotting.input_stream_panels(
            model, inputs, 0, module.input_stream_panels
        ),
    )
    kwargs.update(overrides)
    figure = plotting.build_diagnostic_figure(**kwargs)
    figure.canvas.draw()
    return figure


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


def _axis_limits(figure: Any) -> Dict[str, Any]:
    """The two lag panels' secondary limits, keyed by panel."""
    return {prefix: secondary.get_ylim() for prefix, _panel, secondary in _lag_axes(figure)}


def test_filling_the_panels_delays_with_the_warm_up_did_not_move_the_lag_axis():
    r"""The bug class, driven. The input rows carry $W'_c$ under an attribute named ``delays``, and
    the largest of them is a real number of steps -- so an axis built from the panel rather than
    from the model would be long by $\Delta \max_c W'_c$ and nothing would fail. Compared against a
    hand-computed physical lag rather than against the page's own helper, on the unaligned gate
    where $\delta = 0$."""
    module, batch = _module_and_batch()
    model = module.orig_model
    assert _source_delay_steps(model) == int(model.source_delay_steps) == 0
    figure = _page(module, batch)
    try:
        panels = plotting.input_stream_panels(
            model, module._build_forward_inputs(batch), 0, module.input_stream_panels
        )
        worst = int(np.max(panels[1].delays))
        assert worst > 0, "a flat warm-up would make this vacuous"

        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            assert secondary.get_ylim() == pytest.approx(
                (SECONDS_PER_STEP * low, SECONDS_PER_STEP * high)
            ), prefix
            assert secondary.get_ylim() != pytest.approx(
                (SECONDS_PER_STEP * (low + worst), SECONDS_PER_STEP * (high + worst))
            ), prefix
            assert secondary.get_ylabel() == COMPENSATED_LAG_AXIS_LABEL, prefix
    finally:
        plt.close(figure)


def test_both_panels_carry_the_axis_the_models_own_delay_implies():
    r"""Each panel's primary axis is the lag index $\ell$ and its secondary is $\Delta(\ell + \delta)$
    seconds. The two must be the same map on both panels: the attention map says where the source
    was attended and the KL-by-lag map how much it bought, and they are read together, so two axes
    disagreeing would misalign the only comparison the pair supports.

    Aligned, so $\delta > 0$ and the page's own delay probe -- the read site the recorded bug was
    in -- has a non-zero value to get wrong."""
    module, batch = _module_and_batch(aligned=True)
    figure = _page(module, batch)
    try:
        delay = int(module.orig_model.source_delay_steps)
        assert delay > 0, "an unshifted gate makes this the unaligned test again"
        assert _source_delay_steps(module.orig_model) == delay
        seen = []
        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            assert secondary.get_ylim() == pytest.approx(
                (
                    float(lag_compensated_seconds(low, delay_steps=delay)),
                    float(lag_compensated_seconds(high, delay_steps=delay)),
                )
            ), prefix
            seen.append(secondary.get_ylim())
        assert seen[0] == pytest.approx(seen[1])
    finally:
        plt.close(figure)


def test_a_non_zero_lag_floor_moves_the_mask_and_not_the_axis():
    r"""The second offset, and the honest answer is that there is none.

    ``lag_floor`` restricts **which** lags a step may read -- $\mathbb 1[t - \ell \ge F_u]$, so at
    step $t$ the admissible lags are $\ell \le t - F_u$ -- and leaves what a lag index *means*
    exactly where it was: lag $\ell$ is source step $t - \ell$ at every floor. An axis shifted by the
    floor would therefore report every peak $\Delta F_u$ seconds too long, which is the same failure
    as reading the warm-up as a delay, in the one place where the quantity really does live on the
    lag domain. What the floor does move is asserted beside it, so "the axis did not move" cannot
    pass by the floor doing nothing at all.
    """
    floored, batch = _module_and_batch(lag_floor=_LAG_FLOOR)
    unfloored, _batch = _module_and_batch()

    with_floor = _page(floored, batch)
    try:
        floored_limits = _axis_limits(with_floor)
    finally:
        plt.close(with_floor)

    without_floor = _page(unfloored, batch)
    try:
        unfloored_limits = _axis_limits(without_floor)
    finally:
        plt.close(without_floor)

    for prefix in floored_limits:
        assert floored_limits[prefix] == pytest.approx(unfloored_limits[prefix]), prefix

    # And the floor is not inert: the mask it builds silences the rows below it and the far lags
    # above them, which is the whole of what a floor does.
    length = int(floored.orig_model.sequence_length)
    floored_mask = floored.orig_model.build_lag_mask(length)
    plain_mask = unfloored.orig_model.build_lag_mask(length)
    assert not torch.equal(floored_mask, plain_mask)
    assert not floored_mask[:_LAG_FLOOR].any()
    steps = torch.arange(length)[:, None]
    lags = torch.arange(floored.orig_model.lag_attn.L)[None, :]
    assert torch.equal(floored_mask, plain_mask & (steps - lags >= _LAG_FLOOR))


def test_the_replaced_row_does_not_move_the_lag_axis():
    """The seam owns rows $1$ and $2$ and the two input rows; the lag panels are the builder's.
    Drawing the same model twice -- once through this package's rows and once through a seam that
    draws nothing -- must put the identical axis on both, which is what "the figure's lag axis agrees
    with the metrics'" means once the rows above it are replaceable.

    It is a live risk here rather than a formality: this cell's forecast row is the only one in the
    family drawn on the **raw** grid while every row below it is on the decimated one, so a row that
    pinned its own limits would be the one that moved them."""
    module, batch = _module_and_batch()

    replaced = _page(module, batch)
    try:
        with_rows = _axis_limits(replaced)
    finally:
        plt.close(replaced)

    def _nothing(rows: Any) -> None:
        """A seam that claims its two rows and draws nothing in them."""
        for name in ("raw", "forecast"):
            main, cax = rows.row_axes(name)
            main.set_title(f"placeholder {name}")
            cax.set_visible(False)

    bare = _page(module, batch, forecast_rows=_nothing, input_streams=())
    try:
        without_rows = _axis_limits(bare)
    finally:
        plt.close(bare)

    for prefix in with_rows:
        assert with_rows[prefix] == pytest.approx(without_rows[prefix]), prefix
