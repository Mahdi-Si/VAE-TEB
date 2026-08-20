r"""The figure's lag axis and the model's own reported lag must be the same number.

The raw-signal sibling records the failure this file exists for: two consumers each reached into
the model for the causal input delay $\delta$ under a name of its own guessing, one of those names
did not exist and silently read zero, and at the $120$ s budget the figure's lag axis came out
short by $30$ steps -- two minutes -- against the evaluation's. Both went on producing plausible
numbers.

**This package re-creates that bug class twice over**, which is why the file is here rather than
inherited.

The input rows now fill ``InputStreamPanel.delays`` with $W'_c$, a **warm-up**, through an
attribute whose name says delay. The two quantities are not the same thing and only one of them
belongs on the lag axis: a warm-up is a leading region a channel is not honest in, and it shifts
nothing, whereas a delay means the source memory the attention queries is itself $\delta$ steps
stale. This family applies no delay at all -- its gate is a pure gather -- so the honest axis is
$4\ell$, and a consumer that took the panel's staircase for a delay would report every lag up to
$536$ s too long with nothing failing.

And ``lag_floor`` introduces a second offset on the same axis, which is the more tempting mistake
because it *is* a lag-domain quantity. It is not a shift either: the floor generalises the mask
from $\mathbb 1[t - \ell \ge 0]$ to $\mathbb 1[t - \ell \ge F_u]$, which restricts **which** lags a
step may read, not what a lag index means. Lag $\ell$ is still source step $t - \ell$ at every
floor, so the axis must not move by the floor -- and what must move is the mask.

The secondary axis is read **after a draw**. Matplotlib defers a secondary axis's limits to draw
time, so an assertion made before one passes against the default $(0, 1)$ whatever the transform
is -- which would make this file pass on exactly the bug it is here to catch.
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
    lag_compensated_seconds,
)
from teb_vae.lag_attn_cfs import sample_page  # noqa: E402
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import _source_delay_steps  # noqa: E402

from .conftest import make_stub_batch, make_task, tiny_warmup_kwargs  # noqa: E402

#: The two lag panels' title prefixes, in the order the page lays them out.
_LAG_PANELS = ("Lag attention", r"$\widetilde K_{t,\ell}$")

#: A floor inside the tiny geometry's trained range, so the rows it silences are real rows.
_LAG_FLOOR = 3


def _module_and_batch(**overrides: Any) -> Tuple[Any, Any]:
    """Build this model wrapped in its task, at the tiny warm-up guard.

    Args:
        **overrides: Constructor keywords applied on top of the guarded tiny set.

    Returns:
        ``(task, batch)``.
    """
    module = make_task(model_kwargs=tiny_warmup_kwargs(**overrides))
    return module, make_stub_batch()


def _page(module: Any, batch: Any) -> Any:
    """Draw the whole page through the callback's own seams and force a draw.

    Args:
        module: The task whose net is drawn.
        batch: The batch to draw from.

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
    figure = plotting.build_diagnostic_figure(
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
        forecast_extra_rows=module.forecast_extra_rows,
    )
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


def test_this_family_applies_no_delay_so_the_probe_reads_zero():
    r"""The gate is a pure gather here -- a warm-up masks a leading region and leaves every step at
    its own index -- so there is no staleness to compensate and $4\ell$ is the honest axis. Stated
    as its own assertion because every equality below rests on it, and because a model that had
    acquired a delay would make them all pass while meaning something else."""
    module, _batch = _module_and_batch()
    model = module.orig_model

    assert int(model.source_delay_steps) == _source_delay_steps(model) == 0
    assert model.source_gate is not None, "an ungated model makes this vacuous"
    assert int(model.source_gate.delay.delay_steps.max()) == 0


def test_filling_the_panels_delays_with_the_warm_up_did_not_move_the_lag_axis():
    r"""The bug class, driven. The input rows carry $W'_c$ under an attribute named ``delays``, and
    the largest of them is a real number of steps -- so an axis built from the panel rather than
    from the model would be long by $\Delta \max_c W'_c$ and nothing would fail. Compared against a
    hand-computed physical lag rather than against the page's own helper."""
    module, batch = _module_and_batch()
    model = module.orig_model
    figure = _page(module, batch)
    try:
        panels = sample_page.causal_stream_panels(
            model, module._build_forward_inputs(batch), sample_index=0
        )
        worst = int(np.max(panels[1].delays))
        assert worst > 0, "a flat warm-up would make this vacuous"

        for prefix, panel, secondary in _lag_axes(figure):
            low, high = panel.get_ylim()
            assert secondary.get_ylim() == pytest.approx((4.0 * low, 4.0 * high)), prefix
            assert secondary.get_ylim() != pytest.approx(
                (4.0 * (low + worst), 4.0 * (high + worst))
            ), prefix
            assert secondary.get_ylabel() == COMPENSATED_LAG_AXIS_LABEL, prefix
    finally:
        plt.close(figure)


def test_both_panels_carry_the_axis_the_models_own_delay_implies():
    r"""Each panel's primary axis is the lag index $\ell$ and its secondary is $4(\ell + \delta)$
    seconds. The two must be the same map on both panels: the attention map says where the source
    was attended and the KL-by-lag map how much it bought, and they are read together, so two axes
    disagreeing would misalign the only comparison the pair supports."""
    module, batch = _module_and_batch()
    figure = _page(module, batch)
    try:
        delay = int(module.orig_model.source_delay_steps)
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
    exactly where it was: lag $\ell$ is source step $t - \ell$ at every floor. An axis shifted by
    the floor would therefore report every peak $\Delta F_u$ seconds too long, which is the same
    failure as reading the warm-up as a delay, in the one place where the quantity really does live
    on the lag domain. What the floor does move is asserted beside it, so "the axis did not move"
    cannot pass by the floor doing nothing at all.
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


def test_the_replaced_rows_do_not_move_the_lag_axis():
    """The seams own rows $1$, $2$ and the two input rows; the lag panels are the builder's. Drawing
    the same model twice -- once through this package's rows and once through a seam that draws
    nothing -- must put the identical axis on both, which is what "the figure's lag axis agrees
    with the metrics'" means once the rows above it are replaceable."""
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

    model = module.orig_model
    with torch.no_grad():
        outs = model(*module._build_forward_inputs(batch))
        target, _weight = module._build_raw_target(batch)
        kld_per_dim = model.kld_tensor(
            mu_prior=outs["mu_prior"],
            logvar_prior=outs["logvar_prior"],
            mu_post=outs["mu_post"],
            logvar_post=outs["logvar_post"],
        )
    bare = plotting.build_diagnostic_figure(
        outs=outs, kld_per_dim=kld_per_dim, fhr_raw=target, geometry=model.geometry,
        sample_index=0, epoch=0, guid="SEG000", beta=1.0, scalars={}, up_raw=batch.up,
        normalization_stats=None, delay_steps=_source_delay_steps(model),
        forecast_rows=_nothing, batch=batch,
    )
    bare.canvas.draw()
    try:
        without_rows = _axis_limits(bare)
    finally:
        plt.close(bare)

    for prefix in with_rows:
        assert with_rows[prefix] == pytest.approx(without_rows[prefix]), prefix


# =================================================================================================
# The evaluation pipeline is the third consumer of the same delta
#
# The figure and the model agree above. The evaluation is the consumer that did not exist when
# that agreement was first established, and it is the one whose numbers get quoted: its two
# lag-resolved analyses build their seconds axis from a ``delay_steps`` the run reads off the model
# once and threads through the collection record. A second read under a guessed name is exactly the
# failure this file exists for, so the read site is pinned here rather than only exercised.
# =================================================================================================
def test_the_evaluation_reads_the_delay_off_the_model_and_from_nowhere_else():
    r"""One read site, and it is ``model.source_delay_steps``.

    Pinned by inspecting the runner's source rather than by comparing two numbers that happen to be
    zero on this cell: the whole point is that a *non-zero* delay reaching one consumer and not
    another is invisible, and this family's delay is zero, so a value comparison here would pass
    against a consumer that read nothing at all.
    """
    import inspect

    from teb_vae.lag_attn_cfs.eval import run as run_module

    source = inspect.getsource(run_module)

    assert source.count("int(task.orig_model.source_delay_steps)") == 1
    # And no other attribute name is reached for. ``_source_delay_steps`` is the plotting sibling's
    # accessor and is asserted equal to the model's own above; a second *name* in the runner would
    # be the guessed one.
    #
    # Three mentions, not two, and the third is a comment. The count moved when the alignment gave
    # this cell a SECOND delay-like constant: ``source_delay_steps`` is a stored-step maximum
    # attained by the fastest channel, while the alignment reference is the physical instant every
    # aligned channel reports at a step, and the runner records both. The comment naming the
    # distinction at the read site is worth a unit of this budget; a fourth mention, or a second
    # read expression, is not.
    assert source.count("source_delay_steps") == 3, (
        "the runner must read the delay once and record it once; a further mention is a second "
        "read site, which is how the two reports of one run came to disagree by two minutes"
    )
    assert source.count("source_reference_delay_s") >= 1, (
        "the physical constant must travel beside the stored-step one, or a consumer wanting a lag "
        "in seconds has only the wrong number to reach for"
    )


def test_every_reported_lag_carries_the_maximum_over_channels_flag():
    r"""The source channels are masked individually and the model reports the **maximum**, so every
    lag computed from it is an upper bound. The flag travels beside the numbers rather than being
    stated once elsewhere, because a lag quoted without it reads as exact.

    Asserted on all three emitters at once -- the lag report, and both analyses that read it -- so a
    new emitter that dropped the flag fails here rather than in whichever summary is read first.
    """
    from teb_vae.lag_attn_cfs.eval.analyses import attention as attention_analysis
    from teb_vae.lag_attn_cfs.eval.analyses import lag_kl as lag_kl_analysis
    from teb_vae.lag_attn_cfs.eval.metrics import Aggregate, lag_summary

    module, _batch = _module_and_batch()
    delay = int(module.orig_model.source_delay_steps)
    aggregate = Aggregate(
        overall={},
        lag_profile=[0.5, 0.3, 0.2],
        lag_profile_support_corrected=[0.5, 0.3, 0.2],
        lag_profile_untruncated=[0.5, 0.3, 0.2],
        lag_support=[10.0, 10.0, 10.0],
        attention_profile=[0.5, 0.3, 0.2],
        kld_per_head=[1.0, 1.0],
    )

    report = lag_summary(aggregate, delay_steps=delay)

    assert report["delay_steps"] == delay
    assert report["source_delay_is_max_over_channels"] is True
    # And both analyses carry it through rather than recomputing or dropping it.
    for module_under_test in (lag_kl_analysis, attention_analysis):
        assert "source_delay_is_max_over_channels" in inspect_source(module_under_test)


def inspect_source(module: Any) -> str:
    """The module's source text, for the flag-propagation assertion above."""
    import inspect

    return inspect.getsource(module)
