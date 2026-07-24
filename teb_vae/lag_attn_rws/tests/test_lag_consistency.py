r"""The training figure and the evaluation must report the *same* lag for the same run.

Two consumers add the causal input delay $\delta$ back to an attention lag index: the per-epoch
diagnostic figure and the offline evaluation. They run in different processes, months apart, and
nothing downstream compares them -- so if they disagree, both keep producing plausible numbers and
only a reader who happened to hold a figure beside a ``summary.json`` would ever notice.

They did disagree. Each reached into the model for the delay under a name of its own guessing, and
one of those names did not exist, so it silently read zero: at the $120$ s budget the figure's lag
axis was short by 30 steps -- two minutes -- against the evaluation's. The fix was to give the
model one accessor and have both read it. This file is what keeps that true.
"""
from __future__ import annotations

import torch

from teb_vae.lag_attn_rws.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn_rws.nets.lag_report import lag_compensated_seconds
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws
from teb_vae.lag_attn_rws.plotting import _source_delay_steps

from .conftest import TINY_KWARGS

#: The budget the guard is costed at, and the delay it resolves to.
_BUDGET_S, _EXPECTED_DELAY = 120.0, 30


def _guarded_model() -> SeqVaeLagAttnRws:
    """A tiny model carrying the production reach budget's resolved channel tuples."""
    budget = resolve_stream_budgets(
        {
            "causal_reach_budget_s": _BUDGET_S,
            "use_up_st": True,
            "warmup_period": 30,
            "c_y": 109,
            "c_u": 58,
        }
    )
    torch.manual_seed(0)
    return SeqVaeLagAttnRws(
        **dict(
            TINY_KWARGS,
            sequence_length=64,
            warmup_period=30,
            target_keep_index=budget.target_keep_index,
            target_delays=budget.target_delays,
            source_keep_index=budget.source_keep_index,
            source_delays=budget.source_delays,
        )
    )


def test_the_plotting_callback_reads_the_models_own_delay():
    """The figure's probe must resolve to the model's accessor, not to its zero default."""
    model = _guarded_model()

    assert _source_delay_steps(model) == _EXPECTED_DELAY


def test_the_figure_and_the_evaluation_agree_on_the_delay():
    """The two consumers, side by side. This is the assertion that was false before the model
    grew a single accessor, and it is the only place the two are ever compared."""
    model = _guarded_model()

    # What the evaluation entry point passes to `lag_summary`.
    evaluation_delay = int(model.source_delay_steps)
    # What the plotting callback passes to the figure builder.
    figure_delay = _source_delay_steps(model)

    assert evaluation_delay == figure_delay == _EXPECTED_DELAY
    assert lag_compensated_seconds(5, delay_steps=evaluation_delay) == lag_compensated_seconds(
        5, delay_steps=figure_delay
    )


def test_both_report_zero_for_an_unguarded_model():
    """The shipped default, where the two agreeing is easy -- and where a regression that made
    the guarded case read zero again would otherwise hide."""
    torch.manual_seed(0)
    model = SeqVaeLagAttnRws(**TINY_KWARGS)

    assert int(model.source_delay_steps) == _source_delay_steps(model) == 0
