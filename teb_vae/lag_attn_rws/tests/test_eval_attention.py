r"""The attention: per head, against the ceiling it can reach, and unbiased by truncation.

Three properties, each of which a plausible implementation gets wrong in a way nothing else in
the run would notice.

**The ceiling.** A distribution over $n$ outcomes has entropy at most $\log n$, and at anchor $t$
only $\min(t + 1, L)$ lags exist. At the shipped geometry $60$ of the $240$ trained anchors are
structurally truncated, so $\log L$ is unreachable and a model attending uniformly over
everything available to it reads as *concentrated* when measured against it. The count is
recomputed from the model's own geometry here, so a geometry change moves it rather than
invalidating a number written down.

**The restriction.** The support correction fixes each lag bin's denominator. It cannot fix its
numerator: attention rows are renormalised per anchor, so a truncated anchor pushes the mass that
had no long lag to reach onto the short ones. Only restricting the anchor set removes that, which
is why the restricted argmax exists beside the unrestricted one.

**The heads.** ``kld_per_t_per_head`` was never read at all before this analysis, which discarded
exactly what the head-structured posterior exists to expose. It sums over heads to $K_t$ exactly,
and that is checked on a real run rather than assumed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from teb_vae.lag_attn_rws.eval.figures_seam import figure_filename
from teb_vae.lag_attn_rws.eval.analyses import attention
from teb_vae.lag_attn_rws.eval.collect import geometry_record
from teb_vae.lag_attn_rws.eval.metrics import (
    attainable_lag_entropy,
    attention_entropy,
    untruncated_anchor_mask,
)
from teb_vae.lag_attn.nets.lag_report import lag_compensated_seconds
from teb_vae.lag_attn_rws.nets.model import SeqVaeLagAttnRws

#: What the shipped geometry works out to, asserted against the model rather than hardcoded into
#: the accounting itself: $[30, 270)$ trained anchors, $L = 91$, so anchors $[30, 90)$ are
#: truncated.
_SHIPPED_TRUNCATED_ANCHORS = 60


# =============================================================================
# Truncated support
# =============================================================================
def test_the_truncated_anchor_count_is_sixty_at_the_shipped_geometry(shipped_kwargs) -> None:
    """Recomputed from ``model.geometry`` and the model's own lag width, so a geometry change
    fails this rather than silently leaving a stale number in every summary."""
    model = SeqVaeLagAttnRws(**dict(shipped_kwargs, init_weights=False))

    accounting = attention.truncated_anchor_accounting(
        geometry_record(model.geometry), model.max_lag + 1
    )

    assert accounting["n_trained_anchors"] == 240
    assert accounting["first_untruncated_anchor"] == 90
    assert accounting["n_truncated_anchors"] == _SHIPPED_TRUNCATED_ANCHORS
    assert accounting["truncated_fraction"] == pytest.approx(0.25)


def test_a_lag_window_shorter_than_the_warmup_truncates_nothing() -> None:
    """The other end of the same arithmetic: with $L - 1$ inside the warm-up prefix, every trained
    anchor already sees the whole window and the correction has nothing to correct."""
    accounting = attention.truncated_anchor_accounting(
        {"warmup": 30, "t_valid": 270}, n_lags=9
    )

    assert accounting["n_truncated_anchors"] == 0
    assert accounting["truncated_fraction"] == pytest.approx(0.0)


def test_the_untruncated_mask_is_the_anchors_that_see_every_lag() -> None:
    """$t \\ge L - 1$, and nothing else: the mask the restricted profile is averaged over."""
    mask = untruncated_anchor_mask(seq_len=8, n_lags=4)

    assert mask.tolist() == [False, False, False, True, True, True, True, True]


# =============================================================================
# The two entropies
# =============================================================================
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


def test_the_two_ceilings_are_distinctly_named_and_differ() -> None:
    """The whole point of reporting both: against $\\log L$ a truncated run understates its own
    spread, and a reader normalising by the wrong one draws the opposite conclusion."""
    per_guid = pd.DataFrame(
        {
            "attention_entropy_nats": [1.0, 1.2],
            "attention_entropy_attainable_nats": [2.0, 2.0],
        }
    )

    rows = attention.entropy_rows(per_guid, entropies=[0.9, 1.1, 1.3, 1.5], n_lags=91)

    pooled = rows[0]
    assert pooled["scope"] == "head_averaged"
    assert pooled["ceiling_log_n_lags_nats"] == pytest.approx(math.log(91.0))
    assert pooled["ceiling_attainable_nats"] == pytest.approx(2.0)
    assert pooled["ceiling_attainable_nats"] < pooled["ceiling_log_n_lags_nats"]
    assert (
        pooled["normalised_against_attainable"] > pooled["normalised_against_log_n_lags"]
    )
    # One row per head beside the head-averaged one, never instead of it.
    assert [row["head"] for row in rows[1:]] == [0, 1, 2, 3]


# =============================================================================
# Per-head structure, on a real run
# =============================================================================
def test_the_per_head_kl_sums_over_heads_to_the_headline_kl(evaluated) -> None:
    """The additive decomposition a head-structured posterior buys. Exact rather than
    approximate: the latent groups are head-aligned, so each head's KL is the KL of a group that
    head alone wrote."""
    results = evaluated["summary"]["results"]
    per_head = results["lag"]["kld_per_head"]

    assert len(per_head) == results["lag"]["num_heads"] > 1
    assert sum(per_head) == pytest.approx(
        results["readouts"]["source_conditioned_kl_raw"], rel=1e-6
    )
    assert results["sanity"]["checks"]["per_head_kl_sums_to_kl"]["verdict"] == "pass"


def test_the_per_head_profiles_are_emitted_separately_from_the_head_average(evaluated) -> None:
    """Four heads at four delays average to a flat curve; the head-averaged profile is emitted
    too, and is labelled as an average rather than as "the" profile."""
    lag = evaluated["summary"]["results"]["lag"]
    per_head = lag["attention_lag_profile_per_head"]

    assert len(per_head) == lag["num_heads"]
    assert all(len(profile) == lag["n_lags"] for profile in per_head)
    assert len(lag["attention_lag_profile"]) == lag["n_lags"]
    assert len(lag["attention_entropy_per_head_nats"]) == lag["num_heads"]


def test_the_restricted_argmax_is_emitted_beside_the_unrestricted_one(evaluated) -> None:
    """Both, because the restriction costs a quarter of the anchors at the shipped geometry and
    the unrestricted reading is what a longer profile is compared against."""
    block = evaluated["summary"]["results"]["attention"]

    assert set(block["argmax"]) >= {
        "raw_lag_step", "support_corrected_lag_step", "untruncated_lag_step",
        "restricted_to_anchors_from",
    }
    assert block["source_delay_is_max_over_channels"] is True


def test_the_analysis_writes_its_tables(evaluated) -> None:
    directory = evaluated["results_dir"] / attention.ANALYSIS_DIRNAME
    per_head = pd.read_csv(directory / attention.PER_HEAD_FILENAME)
    lag = evaluated["summary"]["results"]["lag"]

    assert (directory / figure_filename(attention.PROFILE_FIGURE)).is_file()
    assert sorted(set(per_head["head"])) == list(range(lag["num_heads"]))
    assert len(per_head) == lag["num_heads"] * lag["n_lags"]
    assert per_head["compensated_seconds"].tolist()[: lag["n_lags"]] == [
        float(lag_compensated_seconds(index, delay_steps=lag["delay_steps"]))
        for index in range(lag["n_lags"])
    ]


def test_no_heatmap_is_emitted_when_the_attention_was_not_retained(evaluated) -> None:
    """Retention is opt-in and the shipped caps are empty, so the absent figure is silence rather
    than failure -- and the cap that decided it is named in the plan."""
    block = evaluated["summary"]["results"]["attention"]

    assert not (evaluated["results_dir"] / attention.ANALYSIS_DIRNAME
                / figure_filename(attention.HEATMAP_FIGURE)).exists()
    assert block["plan"]["heatmap_cap"] == attention.ATTENTION_CAP
    assert block["plan"]["heatmap_cap_value"] == "absent"


# =============================================================================
# The figures
# =============================================================================
def _lag_block(n_lags: int = 9, num_heads: int = 4) -> dict:
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


def test_the_profile_figure_draws_every_head_and_shades_the_truncated_region() -> None:
    """The shading is structural rather than a finding, and an unshaded figure reads as a model
    that stops attending at long delays."""
    import matplotlib.pyplot as plt

    lag = _lag_block()
    seconds = attention.compensated_seconds_axis(9, delay_steps=0)
    truncation = attention.truncated_anchor_accounting({"warmup": 2, "t_valid": 12}, 9)

    figure = attention.build_profile_figure(
        attention.profile_frame(lag, seconds),
        attention.per_head_frame(lag, seconds),
        delay_steps=0, n_lags=9, truncation=truncation,
    )
    try:
        head_lines = figure.axes[1].get_lines()
        head_x = np.asarray(head_lines[0].get_xdata(), dtype=float)
        shaded = [
            label
            for label in figure.axes[0].get_legend_handles_labels()[1]
            if "truncated" in label
        ]
    finally:
        plt.close(figure)

    assert len(head_lines) == lag["num_heads"]
    assert np.allclose(head_x, seconds)
    assert shaded, "the truncated-support region must be shaded and labelled"


def test_the_heatmap_uses_no_interpolation_and_the_compensated_lag_axis() -> None:
    """``interpolation='none'`` because this is a vector output whose reader indexes a cell: the
    default resampling can put a cell boundary half a cell from where the data says it is."""
    import matplotlib.pyplot as plt

    retained = np.random.default_rng(0).random((2, 12, 4, 9))

    figure = attention.build_heatmap_figure(
        retained, row=0, delay_steps=30, geometry={"warmup": 2}
    )
    try:
        image = figure.axes[0].get_images()[0]
        interpolation = image.get_interpolation()
        extent = image.get_extent()
    finally:
        plt.close(figure)

    assert interpolation == "none"
    # The y extent is the compensated lag axis, which at a delay of 30 starts at 4*30 seconds.
    assert extent[2] == pytest.approx(float(lag_compensated_seconds(0, delay_steps=30)))
    assert extent[3] == pytest.approx(float(lag_compensated_seconds(8, delay_steps=30)))


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
        retained, row=0, delay_steps=0, geometry={"warmup": 2}
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
        _Collection(), tmp_path, delay_steps=0, geometry={"warmup": 2}
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
