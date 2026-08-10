r"""The figure's lag axis and the model's own reported lag must be the same number.

The raw-signal sibling records the failure this file exists for: two consumers each reached into
the model for the causal input delay $\delta$ under a name of its own guessing, one of those names
did not exist, and it silently read zero -- so at the $120$ s budget the figure's lag axis was
short by $30$ steps, two minutes, against the evaluation's. Both went on producing plausible
numbers, and only a reader holding a figure beside a summary would ever have noticed.

**This package has no evaluation pipeline**, deliberately, so its two consumers are the ones that
exist: the model, which reports $\delta$ through one accessor, and the diagnostic page, which
converts a lag index to seconds on both of its lag panels. Nothing else in the tree compares them,
and the conversion is the whole content of the claim a lag panel makes -- a peak at bin $3$ means
nothing until the axis says what second bin $3$ is.

The delay is a **maximum over channels** (the source channels are delayed individually, so no
single $\delta$ describes them all), which is why the page's axis label says *mechanically
compensated* rather than naming an exact physiological lag; the label is asserted here too,
because an axis reading "Lag (s)" is ambiguous between the compensated lag and the uncorrected
sensor one, and those differ by the $20$ s the pipeline already removed.

The secondary axis is read **after a draw**. Matplotlib defers a secondary axis's limits to draw
time, so an assertion made before one passes against the default $(0, 1)$ whatever the transform
is -- which would make this file pass on exactly the bug it is here to catch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from teb_vae.lag_attn.channel_reach import resolve_stream_budgets  # noqa: E402
from teb_vae.lag_attn.nets.lag_report import (  # noqa: E402
    COMPENSATED_LAG_AXIS_LABEL,
    lag_compensated_seconds,
)
from teb_vae.lag_attn_fs.nets.model import SeqVaeLagAttnFs
from teb_vae.lag_attn_fs.task import SeqVaeLagAttnFsTask
from teb_vae.lag_attn_rws import plotting  # noqa: E402
from teb_vae.lag_attn_rws.plotting import _source_delay_steps  # noqa: E402

from .conftest import TASK_HPARAMS, TINY_KWARGS, make_patterned_batch  # noqa: E402

#: The budget the guard is costed at, and the delay it resolves to. Restated rather than imported
#: from the config: what is under test is that the figure reports the delay the model resolved, so
#: a shared constant on both sides would make the comparison circular.
_BUDGET_S, _EXPECTED_DELAY = 120.0, 30

#: A sequence long enough for the guarded warm-up ($30$ steps) to leave trained anchors behind it.
_SEQ_LEN = 64

#: The two lag panels' title prefixes, in the order the page lays them out.
_LAG_PANELS = ("Lag attention", r"$\widetilde K_{t,\ell}$")


def _module(guarded: bool) -> Tuple[Any, Any]:
    """Build this model wrapped in its task, with or without the production reach budget.

    Args:
        guarded: Whether to resolve and apply the $120$ s budget's channel tuples.

    Returns:
        ``(task, batch)`` at the sequence length the guarded warm-up needs.
    """
    kwargs: Dict[str, Any] = dict(
        TINY_KWARGS, sequence_length=_SEQ_LEN, warmup_period=_EXPECTED_DELAY
    )
    if guarded:
        budget = resolve_stream_budgets(
            {
                "causal_reach_budget_s": _BUDGET_S,
                "use_up_st": TINY_KWARGS["use_up_st"],
                "warmup_period": _EXPECTED_DELAY,
                "c_y": TINY_KWARGS["c_y"],
                "c_u": TINY_KWARGS["c_u"],
            }
        )
        kwargs.update(
            target_keep_index=budget.target_keep_index,
            target_delays=budget.target_delays,
            source_keep_index=budget.source_keep_index,
            source_delays=budget.source_delays,
        )
    torch.manual_seed(0)
    model = SeqVaeLagAttnFs(**kwargs)
    task = SeqVaeLagAttnFsTask(model, lr=1e-3, model_kwargs=dict(kwargs), **TASK_HPARAMS)
    task.setup("fit")
    return task, make_patterned_batch(2, _SEQ_LEN)


def _page(module: Any, batch: Any, *, forecast_rows: Any) -> Any:
    """Draw the whole page and force a draw, so every deferred axis has its real limits.

    Args:
        module: The task whose net is drawn.
        batch: The batch to draw from.
        forecast_rows: The row seam, exactly as the callback resolves it.

    Returns:
        A drawn ``Figure``. The caller closes it.
    """
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
    figure = plotting.build_diagnostic_figure(
        outs=outs,
        kld_per_dim=kld_per_dim,
        fhr_raw=target,
        geometry=model.geometry,
        sample_index=0,
        epoch=0,
        guid="rec-0001",
        beta=1.0,
        scalars={},
        up_raw=batch.up,
        normalization_stats=None,
        # The callback's own probe, not a constant: this is the value under test.
        delay_steps=_source_delay_steps(model),
        forecast_rows=forecast_rows,
        batch=batch,
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


def test_the_figures_probe_reads_this_models_own_delay():
    """The subclass inherits the accessor, and inheriting it is the point -- but a model whose
    delay the figure's probe could not resolve would draw a zero-offset axis and say nothing about
    it, which is the exact shape of the historical failure."""
    module, _batch = _module(guarded=True)
    model = module.orig_model

    assert int(model.source_delay_steps) == _source_delay_steps(model) == _EXPECTED_DELAY


def test_both_lag_panels_carry_the_axis_the_models_delay_implies():
    r"""The assertion this file exists for. Each panel's primary axis is the lag index $\ell$ and
    its secondary is $4(\ell + \delta)$ seconds; the two must be the same map on both panels, and
    that map must be the one the *model's* $\delta$ gives -- not a zero-offset one, which is what
    an unresolved delay silently produces."""
    module, batch = _module(guarded=True)
    figure = _page(module, batch, forecast_rows=module.forecast_rows)
    try:
        delay = int(module.orig_model.source_delay_steps)
        assert delay == _EXPECTED_DELAY, "an unguarded model makes every equality below trivial"

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

        # And the two panels agree with each other: the attention map and the KL-by-lag map are
        # read together, one saying where the source was attended and the other how much it
        # bought, so two axes disagreeing would misalign the only comparison the pair supports.
        assert seen[0] == pytest.approx(seen[1])
    finally:
        plt.close(figure)


def test_an_unguarded_model_draws_a_zero_offset_axis_and_that_is_the_honest_one():
    """The other direction, and the one where agreement is easy -- which is why the guarded case
    above is what has to be asserted. Without a reach budget there is no delay to compensate, the
    axis is $4\\ell$, and a regression that made the guarded case read zero again would otherwise
    hide behind this passing."""
    module, batch = _module(guarded=False)
    figure = _page(module, batch, forecast_rows=module.forecast_rows)
    try:
        assert int(module.orig_model.source_delay_steps) == _source_delay_steps(
            module.orig_model
        ) == 0

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


def test_the_feature_rows_do_not_move_the_lag_axis():
    """The seam owns rows $1$ and $2$; the lag panels are rows $6$ and $7$ and are the builder's.
    Drawing the same model twice -- once through this package's rows and once through the raw
    page's default -- must put the identical axis on both, which is what "the figure's lag axis
    agrees with the metrics'" means once the rows underneath it are replaceable.
    """
    module, batch = _module(guarded=True)
    with_feature_rows = _page(module, batch, forecast_rows=module.forecast_rows)
    try:
        replaced = [secondary.get_ylim() for _, _, secondary in _lag_axes(with_feature_rows)]
    finally:
        plt.close(with_feature_rows)

    # The default rows cannot draw a feature target -- they plot it against the raw time axis --
    # so the comparison run supplies a seam that draws nothing at all. What is being compared is
    # the five rows the seam does not own.
    def _nothing(rows: Any) -> None:
        for name in ("raw", "forecast"):
            main, cax = rows.row_axes(name)
            main.set_title(f"placeholder {name}")
            cax.set_visible(False)

    without = _page(module, batch, forecast_rows=_nothing)
    try:
        inherited = [secondary.get_ylim() for _, _, secondary in _lag_axes(without)]
    finally:
        plt.close(without)

    assert replaced == pytest.approx(inherited)


def test_the_task_is_what_binds_the_rows_the_callback_draws():
    """The route, end to end and in one assertion: the callback resolves ``forecast_rows`` off the
    task and the builder draws through it, so a page's rows and its lag axis come from the same
    object and cannot describe two different models."""
    module, _batch = _module(guarded=True)

    assert isinstance(module, SeqVaeLagAttnFsTask)
    assert callable(module.forecast_rows)
    assert getattr(module, "forecast_rows", None) is not None
