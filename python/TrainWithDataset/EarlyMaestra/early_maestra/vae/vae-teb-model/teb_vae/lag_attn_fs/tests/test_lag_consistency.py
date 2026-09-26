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
single $\delta$ describes them all), which is why the page's axis label says *input-delay
compensated* rather than naming an exact physiological lag; the label is asserted here too,
because an axis reading "Lag (s)" does not say whether $\delta$ was added back. The stored
timeline is canonical: no other correction is ever applied to a lag axis.

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
