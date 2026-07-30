r"""The training figure and the evaluation must report the *same* lag for the same run.

Two consumers add the causal input delay $\delta$ back to an attention lag index: the per-epoch
diagnostic figure and the offline evaluation. They run in different processes, months apart, and
nothing downstream compares them -- so if they disagree, both keep producing plausible numbers and
only a reader who happened to hold a figure beside a ``summary.json`` would ever notice.

They did disagree. Each reached into the model for the delay under a name of its own guessing, and
one of those names did not exist, so it silently read zero: at the $120$ s budget the figure's lag
axis was short by 30 steps -- two minutes -- against the evaluation's. The fix was to give the
model one accessor and have both read it. This file is what keeps that true.

The evaluation is now several consumers rather than one -- the lag report the pass assembles, the
per-lag KL analysis and the attention analysis, each drawing an axis -- so the comparison below
covers every one of them against the callback. And because the delay is a **maximum over
channels** (the source channels are delayed individually, so no single $\delta$ describes them
all), every reported lag is an upper bound; the flag saying so must travel with the number rather
than be stated once somewhere else, which the last test here is what enforces.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch

from teb_vae.lag_attn_rws.channel_reach import resolve_stream_budgets
from teb_vae.lag_attn_rws.eval import lag_axis
from teb_vae.lag_attn_rws.eval.analyses import attention as attention_analysis
from teb_vae.lag_attn_rws.eval.analyses import lag_kl as lag_kl_analysis
from teb_vae.lag_attn_rws.eval.metrics import Aggregate, lag_summary
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


# =============================================================================
# Every evaluation consumer, against the same callback
# =============================================================================
#: A lag profile peaked at a lag the delay would move visibly. Four bins is enough: what is being
#: compared is the conversion, not the shape.
_PEAK_LAG = 3
_N_LAGS = 4


def _aggregate_peaked_at(lag: int, n_lags: int = _N_LAGS) -> Aggregate:
    """An aggregate whose every lag profile peaks at one known bin."""
    profile = [1.0 if index == lag else 0.0 for index in range(n_lags)]
    return Aggregate(
        overall={"source_conditioned_kl_raw": 1.0},
        kld_per_dim=[1.0],
        kld_per_head=[0.25, 0.25, 0.25, 0.25],
        lag_profile=list(profile),
        lag_profile_support_corrected=list(profile),
        lag_support=[1.0] * n_lags,
        attention_profile=list(profile),
        attention_profile_support_corrected=list(profile),
        attention_profile_untruncated=list(profile),
        attention_profile_per_head=[value for _ in range(4) for value in profile],
        attention_entropy_per_head=[0.0] * 4,
    )


def test_the_analyses_share_one_axis_builder_rather_than_each_owning_one():
    """Identity, not equality: two implementations that agree today are the exact configuration
    that produced the historical disagreement, and the numeric comparison below would pass on
    both of them right up until one of them changed."""
    assert (
        lag_kl_analysis.compensated_seconds_axis
        is attention_analysis.compensated_seconds_axis
        is lag_axis.compensated_seconds_axis
    )


def test_every_evaluation_consumer_agrees_with_the_callback_at_a_nonzero_delay():
    """The lag report, both analyses' axes, and the training figure, on one guarded model.

    Each of these draws or quotes a lag independently. The comparison is against the *callback's*
    number rather than against a constant, because the failure this file exists for is two
    consumers that each look right on their own.
    """
    model = _guarded_model()
    delay = _source_delay_steps(model)
    expected = float(lag_compensated_seconds(_PEAK_LAG, delay_steps=delay))

    report = lag_summary(_aggregate_peaked_at(_PEAK_LAG), delay_steps=int(model.source_delay_steps))
    kl_axis = lag_kl_analysis.compensated_seconds_axis(_N_LAGS, delay_steps=delay)
    attention_axis = attention_analysis.compensated_seconds_axis(_N_LAGS, delay_steps=delay)

    assert delay == _EXPECTED_DELAY, "an unguarded model would make every equality below trivial"
    assert report["kl_lag_compensated_seconds"] == expected
    assert report["attention_lag_compensated_seconds"] == expected
    assert report["attention_lag_compensated_seconds_untruncated"] == expected
    assert float(kl_axis[_PEAK_LAG]) == expected
    assert float(attention_axis[_PEAK_LAG]) == expected


def _blocks_reporting_a_delay(node: Any, path: str = "results") -> List[str]:
    """Every path in a summary at which a ``delay_steps`` is reported.

    Args:
        node: A summary fragment.
        path: The dotted path reached so far.

    Returns:
        The paths of the dicts carrying a delay.
    """
    found: List[str] = []
    if isinstance(node, dict):
        if "delay_steps" in node:
            found.append(path)
        for key, value in node.items():
            found.extend(_blocks_reporting_a_delay(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_blocks_reporting_a_delay(value, f"{path}[{index}]"))
    return found


def _dig(summary: Dict[str, Any], path: str) -> Dict[str, Any]:
    """Resolve one of :func:`_blocks_reporting_a_delay`'s paths back to its dict."""
    node: Any = summary
    for part in path.split(".")[1:]:
        node = node[part]
    return node


def test_the_max_over_channels_flag_travels_with_every_reported_lag(evaluated):
    """The source channels are delayed individually and the model reports the maximum, so every
    lag derived from it is an upper bound. A lag quoted without that flag reads as exact, and the
    flag stated once in a document beside the run is a flag nobody reads."""
    summary = evaluated["summary"]
    blocks = _blocks_reporting_a_delay(summary["results"])

    assert blocks, "no block reported a delay at all, so this test proved nothing"
    missing = [
        path
        for path in blocks
        if "source_delay_is_max_over_channels" not in _dig(summary["results"], path)
    ]
    assert missing == [], f"{missing} report a lag delay without saying it is a maximum"
