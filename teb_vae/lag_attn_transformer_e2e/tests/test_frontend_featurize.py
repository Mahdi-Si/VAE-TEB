r"""The three-channel featurisation: what a gap looks like, and what it must not look like.

This is the only place in the package where the loader's decimated validity meets the raw signal,
and every one of its failure modes produces a plausible-looking tensor rather than an error.

A gap that arrived as a raw zero would z-score to roughly $-7\sigma$ for the target signal -- an
extreme-bradycardia-looking constant that the low-pass then smears across the following tokens, so
the model would learn a deceleration where there is no data. A gap that arrived as a *normalised*
zero with no mask channel would be indistinguishable from a genuine mid-range sample. A first
difference gated on one endpoint only would inject a step of $\mathcal O(\sigma)$ at the first
valid sample after every gap, which is exactly where a real deceleration would sit.

Each of those is asserted here, and the last test is the negative control the others need: on the
stub batch's planted gap the mask channel must not be identically one, or every criterion above is
being checked against a batch that has no gap in it.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_rws.nets.raw_masks import VALID_THRESHOLD
from teb_vae.lag_attn_transformer_e2e.nets.frontend import FEATURE_CHANNELS, featurize
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    SEQ_LEN,
    STUB_GAP_STEP,
    make_stub_batch,
)

#: Raw samples per decimated step, as the loader writes them. Restated rather than imported from
#: the front end, so a change to the front end's stride cannot silently redefine what this file
#: means by "a step".
RAW_PER_STEP = 16

#: Channel indices of the featurisation, so an assertion says what it is about.
VALUE, MASK, DELTA = 0, 1, 2


def _grid(steps: int = 6, *, gap_at: int = 3) -> tuple:
    """Build a deterministic ``(raw, weight)`` pair with one fully invalid step.

    Args:
        steps: Decimated steps $T$.
        gap_at: The step whose weight is zeroed.

    Returns:
        ``(raw, weight)`` at ``(1, steps * RAW_PER_STEP)`` and ``(1, steps)``.
    """
    raw = torch.arange(1, steps * RAW_PER_STEP + 1, dtype=torch.float32).unsqueeze(0)
    weight = torch.ones(1, steps)
    weight[:, gap_at] = 0.0
    return raw, weight


def test_the_output_is_three_channels_at_the_raw_rate():
    raw, weight = _grid()

    features = featurize(raw, weight)

    assert features.shape == (1, FEATURE_CHANNELS, raw.shape[-1])


def test_an_invalid_step_is_zero_in_every_channel_for_all_of_its_raw_samples():
    """Exactly zero, not small: the value and the difference must carry no trace of a sample the
    loss will not score, and the mask must say so for all sixteen samples rather than for the step
    boundary alone."""
    raw, weight = _grid(gap_at=3)
    span = slice(3 * RAW_PER_STEP, 4 * RAW_PER_STEP)

    features = featurize(raw, weight)

    assert torch.equal(features[0, VALUE, span], torch.zeros(RAW_PER_STEP))
    assert torch.equal(features[0, MASK, span], torch.zeros(RAW_PER_STEP))
    assert torch.equal(features[0, DELTA, span], torch.zeros(RAW_PER_STEP))
    # ...and the surrounding steps are untouched, or the test would pass on an all-zero output.
    assert float(features[0, MASK, : 3 * RAW_PER_STEP].min()) == 1.0
    assert float(features[0, MASK, 4 * RAW_PER_STEP :].min()) == 1.0


def test_a_non_finite_sample_yields_an_all_finite_output():
    """A NaN multiplied by a zero mask is still a NaN, so the neutralisation cannot be a multiply.
    One NaN left in would propagate through the low-pass into every following token."""
    raw, weight = _grid()
    raw[0, 5] = float("nan")
    raw[0, 6] = float("inf")
    raw[0, 7] = float("-inf")

    features = featurize(raw, weight)

    assert bool(torch.isfinite(features).all())
    assert torch.equal(features[0, MASK, 5:8], torch.zeros(3))
    # The step's other samples stay valid: finiteness is per sample, not per step.
    assert float(features[0, MASK, 8]) == 1.0


def test_the_first_difference_is_zero_at_index_zero_and_after_a_gap():
    r"""Both are the same property. The replicate pad makes $\Delta x_0 = 0$; gating on *both*
    endpoints makes the first valid sample after a gap carry no slope, which is where a spurious
    step would be least distinguishable from a real deceleration."""
    raw, weight = _grid(gap_at=3)
    resume = 4 * RAW_PER_STEP

    features = featurize(raw, weight)

    assert float(features[0, DELTA, 0]) == 0.0
    assert float(features[0, DELTA, resume]) == 0.0
    # The very next sample does carry the slope, so the gate is not simply off everywhere.
    assert float(features[0, DELTA, resume + 1]) == pytest.approx(1.0)


def test_a_genuine_normalised_zero_is_distinguishable_from_a_gap():
    """Same value channel, different mask channel. Without the mask channel the model would have to
    treat "no data" and "exactly at the population mean" as the same observation."""
    raw = torch.zeros(1, 2 * RAW_PER_STEP)
    weight = torch.tensor([[1.0, 0.0]])

    features = featurize(raw, weight)

    real, gap = features[0, :, 0], features[0, :, RAW_PER_STEP]
    assert float(real[VALUE]) == float(gap[VALUE]) == 0.0
    assert float(real[MASK]) == 1.0
    assert float(gap[MASK]) == 0.0


def test_a_partially_valid_step_counts_as_invalid():
    """``VALID_THRESHOLD`` is $\\ge 1$, not $> 0$: a partially valid step still contains raw samples
    at roughly $-11\\sigma$, which would dominate a summed 480-sample likelihood. Imported rather
    than restated so this cannot drift from the mask the objective scores against."""
    raw = torch.ones(1, 2 * RAW_PER_STEP)
    weight = torch.tensor([[VALID_THRESHOLD, VALID_THRESHOLD - 0.5]])

    features = featurize(raw, weight)

    assert float(features[0, MASK, 0]) == 1.0
    assert float(features[0, MASK, RAW_PER_STEP]) == 0.0


# ---------------------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, weight, message",
    [
        (torch.zeros(4, 64), torch.zeros(3, 4), "batch axis"),
        (torch.zeros(2, 62), torch.zeros(2, 4), "positive multiple"),
        (torch.zeros(64), torch.zeros(4), "2-D"),
    ],
    ids=["batch-mismatch", "length-mismatch", "wrong-rank"],
)
def test_a_mismatched_pair_is_refused_by_name(raw, weight, message):
    """The expansion is silent when it succeeds, so a mismatched pair would slide the mask against
    the signal rather than raising.

    Only *divisibility* is checked here, because this function derives the ratio from the two
    lengths and has no opinion about what it should be. That the ratio is the front end's own total
    stride is a different claim, and the front end asserts it against itself.
    """
    with pytest.raises(ValueError, match=message):
        featurize(raw, weight)


# ---------------------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------------------
def test_the_stub_batch_actually_exercises_the_masked_path():
    """The control every test above depends on. A fixture whose weight had quietly become uniform
    would leave all of them green while testing none of the gap behaviour."""
    batch = make_stub_batch(BATCH, SEQ_LEN)

    features = featurize(batch.fhr, batch.weight)

    assert not bool((features[:, MASK] == 1.0).all())
    span = slice(STUB_GAP_STEP * RAW_PER_STEP, (STUB_GAP_STEP + 1) * RAW_PER_STEP)
    assert torch.equal(features[:, MASK, span], torch.zeros(BATCH, RAW_PER_STEP))
