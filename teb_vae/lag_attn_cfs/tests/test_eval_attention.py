r"""The attention: per head, against the ceiling it can reach, and unbiased by truncation.

Three properties, each of which a plausible implementation gets wrong in a way nothing else in
the run would notice.

**The ceiling, and why it is measured here rather than assumed.** A distribution over $n$ outcomes
has entropy at most $\log n$, and at anchor $t$ only $\min(t + 1, L)$ lags exist. On the raw cells
that bites: their trained anchors start at $30$ against $L = 91$, so a quarter are structurally
truncated and $\log L$ is unreachable. **On this cell it does not**, because the anchor floor is
$133$ and every scored anchor sees the whole window -- so the attainable ceiling is exactly
$\log L$. That is a consequence of $F \ge L - 1$, not of the target domain, so the equality is
asserted **conditionally on preflight's measured margin** and a constructed low-floor case is
asserted to break it.

**The restriction.** The support correction fixes each lag bin's denominator. It cannot fix its
numerator: attention rows are renormalised per anchor, so a truncated anchor pushes the mass that
had no long lag to reach onto the short ones. Only restricting the anchor set removes that, which
is why the restricted argmax exists beside the unrestricted one -- and it stays emitted here even
though the restriction currently costs nothing, because an arm can make it cost something.

**The heads.** ``kld_per_t_per_head`` sums over heads to $K_t$ exactly, and that is checked on a
real run rather than assumed. Averaging the heads before profiling would discard exactly what the
head-structured posterior exists to expose.

Per the fixture rule in ``test_eval_fixtures.py``: nothing below asserts where an attention peak
lands. What is asserted is geometry, denominators, shapes and the caveat that travels with them.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_cfs.eval.figures_seam import figure_filename
from teb_vae.lag_attn_cfs.eval import lag_axis
from teb_vae.lag_attn_cfs.eval.analyses import attention
from teb_vae.lag_attn_cfs.eval.metrics import (
    attainable_lag_entropy,
    attention_entropy,
    untruncated_anchor_mask,
)
from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds, SECONDS_PER_STEP

from .conftest import SHIPPED_HORIZON, SHIPPED_SEQUENCE_LENGTH, SHIPPED_WARMUP_PERIOD

#: The shipped lag window: ``max_lag: 90`` searched lags plus lag $0$.
_SHIPPED_N_LAGS = 91

#: The shipped decoded anchor range, from the config's own geometry: $[F, T - H)$.
_SHIPPED_T_VALID = SHIPPED_SEQUENCE_LENGTH - SHIPPED_HORIZON

#: The margin preflight measures at that geometry: $133 - 90 - 0$.
_SHIPPED_MARGIN = SHIPPED_WARMUP_PERIOD - (_SHIPPED_N_LAGS - 1)


def _shipped_geometry() -> Dict[str, Any]:
    """The collection record's geometry block at the shipped configuration."""
    return {"anchor_floor": SHIPPED_WARMUP_PERIOD, "t_valid": _SHIPPED_T_VALID}


def _recorded(margin: int) -> Dict[str, Any]:
    """What ``read_lag_support`` returns for a run at a given measured margin."""
    return {
        "measured": True,
        "lag_support_margin_steps": int(margin),
        "every_lag_valid_at_every_anchor": bool(margin >= 0),
    }


# =================================================================================================
# Truncated support
# =================================================================================================
def test_no_decoded_anchor_is_truncated_at_the_shipped_geometry() -> None:
    r"""$F = 134$ against $L - 1 = 90$, so the correction has nothing to correct.

    Derived from the shipped configuration's own numbers rather than written down, so a
    ``sweep_floor_*`` arm moves the count instead of leaving a stale zero in every summary.

    The count is $136$ rather than $137$, and the one anchor is what the channel alignment costs:
    unaligned the floor is $B - 1 = 133$, because a forecast at anchor $t$ reads target time
    $t + 1$ at the earliest; aligned, a channel is honest at $W'_c + d_c$ and the floor must
    additionally clear $\max_c(W'_c + d_c) = B$ on both streams.
    """
    accounting = attention.truncated_anchor_accounting(_shipped_geometry(), _SHIPPED_N_LAGS)

    assert _SHIPPED_MARGIN >= 0
    assert accounting["decoded_anchor_range"] == [SHIPPED_WARMUP_PERIOD, _SHIPPED_T_VALID]
    assert accounting["n_decoded_anchors"] == 136
    assert accounting["first_untruncated_anchor"] == _SHIPPED_N_LAGS - 1
    assert accounting["n_truncated_anchors"] == 0
    assert accounting["truncated_fraction"] == pytest.approx(0.0)


def test_a_floor_below_the_lag_window_truncates_anchors_again() -> None:
    """Non-vacuity for the zero above. A count that were always zero would say nothing about the
    geometry, and this is the arm that reintroduces the bias the corrections exist to remove."""
    accounting = attention.truncated_anchor_accounting(
        {"anchor_floor": 30, "t_valid": 270}, _SHIPPED_N_LAGS
    )

    assert accounting["n_truncated_anchors"] == 60
    assert accounting["truncated_fraction"] == pytest.approx(0.25)


def test_the_untruncated_mask_is_the_anchors_that_see_every_lag() -> None:
    """$t \\ge L - 1$, and nothing else: the mask the restricted profile is averaged over."""
    mask = untruncated_anchor_mask(seq_len=8, n_lags=4)

    assert mask.tolist() == [False, False, False, True, True, True, True, True]


# =================================================================================================
# The two entropies
# =================================================================================================
def test_the_attainable_ceiling_is_log_of_the_lags_that_exist() -> None:
    r"""$\log\min(t+1, L)$, which is $\log L$ only from $t = L - 1$ on."""
    ceiling = attainable_lag_entropy(seq_len=5, n_lags=3)

    assert ceiling.tolist() == pytest.approx(
        [math.log(1.0), math.log(2.0), math.log(3.0), math.log(3.0), math.log(3.0)]
    )


def test_a_uniform_attention_reaches_the_entropy_of_its_own_support() -> None:
    """The known answer that pins the entropy arithmetic, including the $0 \\log 0$ convention:
    ``entmax15`` assigns exact zeros and the causal mask assigns more of them, so a naive
    $p \\log p$ produces ``nan`` on the majority of rows rather than a number."""
    weights = torch.zeros(1, 1, 1, 8)
    weights[..., :4] = 0.25

    entropy = attention_entropy(weights)

    assert float(entropy) == pytest.approx(math.log(4.0))


def test_a_concentrated_attention_has_zero_entropy() -> None:
    weights = torch.zeros(1, 1, 1, 8)
    weights[..., 3] = 1.0

    assert float(attention_entropy(weights)) == pytest.approx(0.0)


def test_the_two_ceilings_are_distinctly_named_and_can_differ() -> None:
    """The whole point of reporting both: against $\\log L$ a truncated run understates its own
    spread, and a reader normalising by the wrong one draws the opposite conclusion. Constructed
    truncated, because on this cell's own geometry the two coincide."""
    per_guid = pd.DataFrame(
        {
            "attention_entropy_nats": [1.0, 1.2],
            "attention_entropy_attainable_nats": [2.0, 2.0],
        }
    )

    rows = attention.entropy_rows(per_guid, entropies=[0.9, 1.1, 1.3, 1.5], n_lags=_SHIPPED_N_LAGS)

    pooled = rows[0]
    assert pooled["scope"] == "head_averaged"
    assert pooled["ceiling_log_n_lags_nats"] == pytest.approx(math.log(float(_SHIPPED_N_LAGS)))
    assert pooled["ceiling_attainable_nats"] == pytest.approx(2.0)
    assert pooled["ceiling_attainable_nats"] < pooled["ceiling_log_n_lags_nats"]
    assert pooled["normalised_against_attainable"] > pooled["normalised_against_log_n_lags"]
    # One row per head beside the head-averaged one, never instead of it.
    assert [row["head"] for row in rows[1:]] == [0, 1, 2, 3]


# =================================================================================================
# The measured ceiling: three readings of one property
# =================================================================================================
def _entropy_rows_at(attainable: float, n_lags: int = _SHIPPED_N_LAGS):
    """Entropy rows whose attainable ceiling is exactly the value asked for."""
    per_guid = pd.DataFrame(
        {
            "attention_entropy_nats": [1.0, 1.0],
            "attention_entropy_attainable_nats": [attainable, attainable],
        }
    )
    return attention.entropy_rows(per_guid, entropies=[], n_lags=n_lags)


def test_at_a_non_negative_margin_the_ceiling_is_measured_equal_to_log_l() -> None:
    r"""The simplification $F \ge L - 1$ buys, asserted as a **measurement** conditional on the
    margin rather than as a property of the domain."""
    rows = _entropy_rows_at(math.log(float(_SHIPPED_N_LAGS)))
    truncation = attention.truncated_anchor_accounting(_shipped_geometry(), _SHIPPED_N_LAGS)

    ceiling = attention.measured_ceiling(rows, truncation, _recorded(_SHIPPED_MARGIN))

    assert ceiling["lag_support_margin_steps"] >= 0
    assert ceiling["ceiling_equals_log_n_lags"] is True
    assert ceiling["ceiling_abs_difference"] <= attention.CEILING_TOLERANCE
    assert ceiling["untruncated_by_geometry_record"] is True
    assert ceiling["computed_and_observed_agree"] is True


def test_at_a_negative_margin_the_ceiling_is_measured_below_log_l() -> None:
    """Non-vacuity for the case above: a test that only ever saw the shipped floor would pass on an
    implementation that returned ``True`` unconditionally."""
    truncated_geometry = {"anchor_floor": 30, "t_valid": 270}
    rows = _entropy_rows_at(math.log(float(_SHIPPED_N_LAGS)) - 0.3)
    truncation = attention.truncated_anchor_accounting(truncated_geometry, _SHIPPED_N_LAGS)

    ceiling = attention.measured_ceiling(rows, truncation, _recorded(30 - 90))

    assert ceiling["lag_support_margin_steps"] < 0
    assert ceiling["ceiling_equals_log_n_lags"] is False
    assert ceiling["ceiling_abs_difference"] > attention.CEILING_TOLERANCE
    assert ceiling["untruncated_by_geometry_record"] is False
    assert ceiling["computed_and_observed_agree"] is True


def test_three_readings_that_disagree_are_reported_as_disagreeing() -> None:
    """The failure no other number would show: preflight described one geometry, the collection
    record another, and the accumulated entropies a third. Recorded rather than raised."""
    rows = _entropy_rows_at(math.log(float(_SHIPPED_N_LAGS)) - 0.3)
    truncation = attention.truncated_anchor_accounting(_shipped_geometry(), _SHIPPED_N_LAGS)

    ceiling = attention.measured_ceiling(rows, truncation, _recorded(_SHIPPED_MARGIN))

    assert ceiling["computed_and_observed_agree"] is False


def test_an_unmeasured_margin_leaves_the_comparison_undecided() -> None:
    """A run whose margin nobody measured must not report an agreement it cannot have checked."""
    rows = _entropy_rows_at(math.log(float(_SHIPPED_N_LAGS)))
    truncation = attention.truncated_anchor_accounting(_shipped_geometry(), _SHIPPED_N_LAGS)

    ceiling = attention.measured_ceiling(
        rows, truncation, dict(lag_axis.UNMEASURED_LAG_SUPPORT)
    )

    assert ceiling["measured"] is False
    assert ceiling["computed_and_observed_agree"] is None
    # What the run itself measured is still reported: that half needs no preflight record.
    assert ceiling["ceiling_equals_log_n_lags"] is True


# =================================================================================================
# The figures
# =================================================================================================
def _lag_block(n_lags: int = 9, num_heads: int = 4) -> Dict[str, Any]:
    """A lag block shaped like the pass's, with a distinguishable profile per head."""
    base = np.linspace(1.0, 0.0, n_lags)
    return {
        "attention_lag_profile": (base / base.sum()).tolist(),
        "attention_lag_profile_support_corrected": (base / base.sum()).tolist(),
        "attention_lag_profile_untruncated": (base[::-1] / base.sum()).tolist(),
        "attention_lag_profile_per_head": [
            np.roll(base / base.sum(), head).tolist() for head in range(num_heads)
        ],
        "kld_per_head": [1.0] * num_heads,
        "n_lags": n_lags,
        "num_heads": num_heads,
    }


def test_the_profile_figure_draws_every_head_and_shades_a_truncated_region() -> None:
    """The shading is structural rather than a finding, and an unshaded figure would read as a
    model that stops attending at long delays. Constructed truncated: at the shipped floor there is
    no such region and the honest figure has no shading at all."""
    import matplotlib.pyplot as plt

    lag = _lag_block()
    seconds = attention.compensated_seconds_axis(9, delay_steps=0)
    truncation = attention.truncated_anchor_accounting(
        {"anchor_floor": 2, "t_valid": 12}, 9
    )

    figure = attention.build_profile_figure(
        attention.profile_frame(lag, seconds),
        attention.per_head_frame(lag, seconds),
        delay_steps=0, n_lags=9, truncation=truncation,
    )
    try:
        head_lines = figure.axes[1].get_lines()
        head_x = np.asarray(head_lines[0].get_xdata(), dtype=float)
        # The shading's own label, not the untruncated *curve*'s: both contain "truncated" and
        # only one of them says anything about a shaded span.
        shaded = [
            label
            for label in figure.axes[0].get_legend_handles_labels()[1]
            if label.startswith("lags truncated at")
        ]
        label = figure.axes[0].get_xlabel()
    finally:
        plt.close(figure)

    assert truncation["n_truncated_anchors"] > 0
    assert len(head_lines) == lag["num_heads"]
    assert np.allclose(head_x, seconds)
    assert shaded, "the truncated-support region must be shaded and labelled"
    assert label == lag_axis.COEFFICIENT_LAG_AXIS_LABEL


def test_an_untruncated_geometry_shades_nothing() -> None:
    """The other branch, and the one this cell actually runs: shading a region that does not exist
    would tell a reader the long lags were unreachable when every anchor saw them."""
    import matplotlib.pyplot as plt

    lag = _lag_block()
    seconds = attention.compensated_seconds_axis(9, delay_steps=0)
    truncation = attention.truncated_anchor_accounting(_shipped_geometry(), 9)

    figure = attention.build_profile_figure(
        attention.profile_frame(lag, seconds),
        attention.per_head_frame(lag, seconds),
        delay_steps=0, n_lags=9, truncation=truncation,
    )
    try:
        # The shading's own label, not the untruncated *curve*'s: both contain "truncated" and
        # only one of them says anything about a shaded span.
        shaded = [
            label
            for label in figure.axes[0].get_legend_handles_labels()[1]
            if label.startswith("lags truncated at")
        ]
    finally:
        plt.close(figure)

    assert truncation["n_truncated_anchors"] == 0
    assert shaded == []


def test_both_figures_carry_the_group_delay_caveat() -> None:
    """A figure is the artifact most likely to be lifted out of a run directory and shown alone,
    and an attention peak read off one without this beside it is read as a physiological delay."""
    import matplotlib.pyplot as plt

    lag = _lag_block()
    seconds = attention.compensated_seconds_axis(9, delay_steps=0)
    truncation = attention.truncated_anchor_accounting(_shipped_geometry(), 9)

    profile_figure = attention.build_profile_figure(
        attention.profile_frame(lag, seconds),
        attention.per_head_frame(lag, seconds),
        delay_steps=0, n_lags=9, truncation=truncation,
    )
    heatmap_figure = attention.build_heatmap_figure(
        np.random.default_rng(0).random((1, 12, 4, 9)),
        row=0, delay_steps=0, geometry=_shipped_geometry(),
    )
    try:
        profile_texts = [artist.get_text() for artist in profile_figure.texts]
        heatmap_texts = [artist.get_text() for artist in heatmap_figure.texts]
    finally:
        plt.close(profile_figure)
        plt.close(heatmap_figure)

    assert lag_axis.GROUP_DELAY_CAVEAT in profile_texts
    assert lag_axis.GROUP_DELAY_CAVEAT in heatmap_texts


def test_the_heatmap_uses_no_interpolation_and_the_coefficient_time_axis() -> None:
    """``interpolation='none'`` because this is a vector output whose reader indexes a cell: the
    default resampling can put a cell boundary half a cell from where the data says it is."""
    import matplotlib.pyplot as plt

    retained = np.random.default_rng(0).random((2, 12, 4, 9))

    figure = attention.build_heatmap_figure(
        retained, row=0, delay_steps=30, geometry={"anchor_floor": 2}
    )
    try:
        image = figure.axes[0].get_images()[0]
        interpolation = image.get_interpolation()
        extent = image.get_extent()
        label = figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)

    assert interpolation == "none"
    # The y extent is the compensated lag axis, which at a delay of 30 starts at 4*30 seconds --
    # spanning the BINS, half a step past the first and the last lag centre, so the L rows are
    # not squeezed into L - 1 steps.
    half = SECONDS_PER_STEP / 2.0
    assert extent[2] == pytest.approx(float(lag_compensated_seconds(0, delay_steps=30)) - half)
    assert extent[3] == pytest.approx(float(lag_compensated_seconds(8, delay_steps=30)) + half)
    assert label == lag_axis.COEFFICIENT_LAG_AXIS_LABEL


def test_the_heatmaps_lag_axis_is_not_upside_down() -> None:
    """The shared panel draws with ``origin='upper'``, so data row 0 lands at the *top* of the
    extent. Handed the lag axis unreversed, lag $0$ would sit against the largest second on the
    label -- an inversion that no extent assertion can see and that a reader has no way to catch,
    because a heatmap of attention looks plausible either way up.
    """
    import matplotlib.pyplot as plt

    # Attention entirely at lag 0, and nowhere else.
    retained = np.zeros((1, 12, 4, 9))
    retained[..., 0] = 1.0

    figure = attention.build_heatmap_figure(
        retained, row=0, delay_steps=0, geometry={"anchor_floor": 2}
    )
    try:
        drawn = np.asarray(figure.axes[0].get_images()[0].get_array(), dtype=float)
    finally:
        plt.close(figure)

    # The bottom row of the drawn field is where the extent's smallest second is, so it must be
    # the lag-0 row -- the one carrying all the attention.
    assert drawn[-1].tolist() == [1.0] * 12
    assert drawn[0].tolist() == [0.0] * 12


def test_the_heatmap_is_emitted_when_the_attention_was_retained(tmp_path) -> None:
    """The other half of the opt-in rule: with the tensor retained, the figure is written.

    Driven with synthetic retained arrays rather than by running a capped pass, because what is
    being checked is the emission path -- a figure guarded by a condition nothing in the default
    suite satisfies is a figure that is never drawn until a production run needs it.
    """

    class _Collection:
        retained = {attention.ATTENTION_TENSOR: np.random.default_rng(1).random((1, 12, 4, 9))}

    written = attention._emit_heatmap(
        _Collection(), tmp_path, delay_steps=0, geometry=_shipped_geometry()
    )

    assert written == figure_filename(attention.HEATMAP_FIGURE)
    assert (tmp_path / figure_filename(attention.HEATMAP_FIGURE)).is_file()


def test_an_empty_pass_draws_the_empty_note_rather_than_raising() -> None:
    """An analysis that legitimately scored nothing reaches its figure at the end of a run that
    has already cost hours."""
    import matplotlib.pyplot as plt

    empty = {"attention_lag_profile": [], "attention_lag_profile_per_head": []}
    seconds = attention.compensated_seconds_axis(0, delay_steps=0)

    figure = attention.build_profile_figure(
        attention.profile_frame(empty, seconds),
        attention.per_head_frame(empty, seconds),
        delay_steps=0, n_lags=0, truncation=attention.truncated_anchor_accounting({}, 0),
    )
    try:
        texts = [text.get_text() for text in figure.axes[0].texts]
    finally:
        plt.close(figure)

    assert texts, "an empty panel must say so"


# =================================================================================================
# Per-head structure, on a real run
# =================================================================================================
@pytest.mark.slow
def test_the_per_head_kl_sums_over_heads_to_the_headline_kl(collected_run) -> None:
    """The additive decomposition a head-structured posterior buys. Exact rather than
    approximate: the latent groups are head-aligned, so each head's KL is the KL of a group that
    head alone wrote."""
    results = collected_run["summary"]["results"]
    per_head = results["lag"]["kld_per_head"]

    assert len(per_head) == results["lag"]["num_heads"] > 1
    assert sum(per_head) == pytest.approx(
        results["readouts"]["source_conditioned_kl_raw"], rel=1e-6
    )
    assert results["sanity"]["checks"]["per_head_kl_sums_to_kl"]["verdict"] == "pass"


@pytest.mark.slow
def test_the_per_head_profiles_are_emitted_separately_from_the_head_average(collected_run) -> None:
    """Four heads at four delays average to a flat curve; the head-averaged profile is emitted
    too, and is labelled as an average rather than as "the" profile."""
    lag = collected_run["summary"]["results"]["lag"]
    per_head = lag["attention_lag_profile_per_head"]

    assert len(per_head) == lag["num_heads"]
    assert all(len(profile) == lag["n_lags"] for profile in per_head)
    assert len(lag["attention_lag_profile"]) == lag["n_lags"]
    assert len(lag["attention_entropy_per_head_nats"]) == lag["num_heads"]


@pytest.mark.slow
def test_the_restricted_argmax_is_emitted_beside_the_unrestricted_one(collected_run) -> None:
    """All three, even where the restriction currently costs nothing: an arm that lowered the floor
    would make it cost something, and a summary that had stopped emitting it would be unreadable
    against the ones that did."""
    block = collected_run["summary"]["results"]["attention"]

    assert set(block["argmax"]) >= {
        "raw_lag_step", "support_corrected_lag_step", "untruncated_lag_step",
        "restricted_to_anchors_from",
    }
    assert block["source_delay_is_max_over_channels"] is True
    assert block["axis_caveat"] == lag_axis.GROUP_DELAY_CAVEAT


@pytest.mark.slow
def test_the_measured_ceiling_agrees_with_the_recorded_margin(collected_run) -> None:
    r"""The three readings, compared on a real run. The tiny fixture's geometry is not the shipped
    one, so what is asserted is the *agreement*, never a particular margin."""
    ceiling = collected_run["summary"]["results"]["attention"]["ceiling"]

    assert ceiling["measured"] is True
    assert ceiling["computed_and_observed_agree"] is True
    if ceiling["lag_support_margin_steps"] >= 0:
        assert ceiling["ceiling_equals_log_n_lags"] is True
        assert ceiling["untruncated_by_geometry_record"] is True
    else:
        assert ceiling["ceiling_equals_log_n_lags"] is False


@pytest.mark.slow
def test_the_analysis_writes_its_tables(collected_run) -> None:
    directory = Path(collected_run["results_dir"]) / attention.ANALYSIS_DIRNAME
    per_head = pd.read_csv(directory / attention.PER_HEAD_FILENAME)
    lag = collected_run["summary"]["results"]["lag"]

    assert (directory / figure_filename(attention.PROFILE_FIGURE)).is_file()
    assert sorted(set(per_head["head"])) == list(range(lag["num_heads"]))
    assert len(per_head) == lag["num_heads"] * lag["n_lags"]
    assert per_head["compensated_seconds"].tolist()[: lag["n_lags"]] == [
        float(lag_compensated_seconds(index, delay_steps=lag["delay_steps"]))
        for index in range(lag["n_lags"])
    ]


@pytest.mark.slow
def test_the_heatmap_is_emitted_because_the_shipped_delta_retains_the_attention(
    collected_run,
) -> None:
    """Retention is opt-in, and this package's override delta **sets** ``caps.attention`` rather
    than leaving it empty -- deliberately, so a stock run emits the complete artifact set. So the
    heatmap exists here, and the cap that decided it is named in the plan rather than left to be
    inferred from the figure's presence.

    The sibling ships the opposite default and its test asserts the absence; carrying that
    assertion across would have passed only while this cap was unset.
    """
    block = collected_run["summary"]["results"]["attention"]

    assert block["plan"]["heatmap_cap"] == attention.ATTENTION_CAP
    assert block["plan"]["heatmap_cap_value"] not in (None, "absent")
    assert (
        Path(collected_run["results_dir"]) / attention.ANALYSIS_DIRNAME
        / figure_filename(attention.HEATMAP_FIGURE)
    ).is_file()
