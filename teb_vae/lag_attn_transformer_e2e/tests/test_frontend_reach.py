r"""How far back the front end reads, what refuses when that is too far, and the probe that proves
the number is not merely arithmetic somebody wrote down.

The reach matters for one reason. Every convolution in the stack is zero-padded on the left, so the
first ``reach - 1`` raw samples of a segment produce output that is partly a picture of the padding
rather than of the signal. The model's warm-up prefix already excludes an initial band of anchors
from the loss, so the front end is safe exactly while its reach fits inside
``warmup_period * raw_per_step``. Beyond that a *trained* anchor reads the transient, and nothing
downstream can tell that apart from a real feature.

The reported number is accumulated from the **built** modules rather than recomputed from the
constructor arguments, so it cannot disagree with the stack that produced it. That still leaves the
accumulation itself untested, which is what the probe at the bottom is for: it perturbs one raw
sample just outside the claimed support and requires the token to be bitwise unmoved.

The probe pins the **safety** claim, not tightness. Asserting that a perturbation at
$n - R + 1$ *does* move the token would pin the bound as exact, which nothing requires and which
would break the first time a kernel change made the formula conservative.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    build_frontend,
)

#: The production schedule's reach, in raw samples, measured from the built stack. $322$ samples is
#: $80.5$ s at $4$ Hz, against a budget of $30 \times 16 = 480$ ($120$ s).
SHIPPED_REACH_SAMPLES = 322

#: The smoke schedule's reach, against a budget of $6 \times 16 = 96$.
TINY_REACH_SAMPLES = 94


def _budget(kwargs: dict) -> int:
    """The reach budget a geometry implies: ``warmup_period * raw_per_step`` raw samples."""
    return int(kwargs["warmup_period"]) * int(kwargs["raw_per_step"])


# ---------------------------------------------------------------------------------------
# The pinned numbers
# ---------------------------------------------------------------------------------------
def test_the_production_reach_is_pinned_and_fits_its_budget():
    net = build_frontend(SHIPPED_KWARGS)

    assert net.reach_samples == SHIPPED_REACH_SAMPLES
    assert net.reach_samples < _budget(SHIPPED_KWARGS) == 480


def test_the_smoke_reach_is_pinned_and_fits_its_budget():
    """Tighter than production by design: the smoke geometry raises ``warmup_period`` to $6$ purely
    to give a four-stage stride-2 cascade room, and the margin left is what a wider anti-alias
    filter would spend."""
    net = build_frontend(TINY_KWARGS)

    assert net.reach_samples == TINY_REACH_SAMPLES
    assert net.reach_samples < _budget(TINY_KWARGS) == 96


def test_the_reach_is_a_count_that_grows_with_depth():
    """A count, matching the ``receptive_field`` convention of the blocks it is accumulated from, so
    a reach of $1$ would mean "this sample only". Each stage strictly extends it, which is what
    makes a dead stage -- one whose kernel collapsed to a single tap -- visible here rather than
    only in a training curve."""
    net = build_frontend(SHIPPED_KWARGS)

    per_stage = net.stage_reach_samples

    assert len(per_stage) == len(net.stage_modules)
    assert per_stage[-1] == net.reach_samples
    assert all(later > earlier for earlier, later in zip(per_stage, per_stage[1:]))
    assert per_stage[0] > 1


def test_a_wider_kernel_costs_more_reach_at_a_deeper_stage():
    """The stride weighting is the non-obvious half of the arithmetic: a kernel at stage $4$ costs
    $8\\times$ what the same kernel costs at stage $1$, because each of its taps spans eight raw
    samples. A formula that summed the kernels flat would pass every other test in this file."""
    early = build_frontend(TINY_KWARGS, kernels=(7, 3, 3, 3), reach_budget=10_000)
    late = build_frontend(TINY_KWARGS, kernels=(3, 3, 3, 7), reach_budget=10_000)
    base = build_frontend(TINY_KWARGS, kernels=(3, 3, 3, 3), reach_budget=10_000)

    assert early.reach_samples - base.reach_samples == 4
    assert late.reach_samples - base.reach_samples == 4 * 8


# ---------------------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------------------
def test_an_over_wide_schedule_is_refused_at_construction_naming_both_numbers():
    """At construction, not at the first forward: a front end that reaches past its warm-up is a
    geometry error, and discovering it hours into a run costs the run."""
    with pytest.raises(ValueError, match="front end reaches") as excinfo:
        build_frontend(TINY_KWARGS, kernels=(5, 3, 3, 33))

    message = str(excinfo.value)
    assert str(_budget(TINY_KWARGS)) in message
    assert "warmup_period" in message


def test_the_budget_boundary_is_inclusive():
    """A reach exactly equal to the budget is legal: the budget counts the raw samples a warm-up
    anchor covers, and a stack reaching exactly that far reads no sample before the segment starts.
    An off-by-one here would reject a legal geometry with a message about a leak that is not there.
    """
    exact = build_frontend(TINY_KWARGS, reach_budget=TINY_REACH_SAMPLES)

    assert exact.reach_samples == TINY_REACH_SAMPLES
    with pytest.raises(ValueError, match="front end reaches"):
        build_frontend(TINY_KWARGS, reach_budget=TINY_REACH_SAMPLES - 1)


# ---------------------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("token", [8, 12], ids=lambda t: f"t={t}")
def test_a_sample_just_outside_the_claimed_support_moves_nothing(token):
    r"""Token $t$'s causal endpoint is $n = s(t+1) - 1$ and its claimed support is
    $[n - R + 1,\, n]$. Perturbing raw sample $n - R$ must leave the token **bitwise** identical;
    perturbing $n$ must move it, which is the half that shows the probe reached the module at all.

    Driven in float64 with a large amplitude, because a float32 threshold at this boundary would be
    a statement about round-off rather than about reach.
    """
    net = build_frontend(TINY_KWARGS).double()
    stride, reach = net.total_stride, net.reach_samples
    endpoint = stride * (token + 1) - 1
    outside = endpoint - reach
    assert outside >= 0, f"token {token} does not have {reach} raw samples of history before it"

    raw = torch.randn(2, SEQ_LEN * stride, dtype=torch.float64)
    weight = torch.ones(2, SEQ_LEN, dtype=torch.float64)
    with torch.no_grad():
        reference = net(raw, weight)
        moved_outside = net(_perturb(raw, outside), weight)
        moved_inside = net(_perturb(raw, endpoint), weight)

    assert torch.equal(reference[:, token], moved_outside[:, token]), (
        f"raw sample {outside} is {reach} samples before token {token}'s endpoint {endpoint}, "
        f"outside the claimed support, yet it moved the token"
    )
    assert not torch.equal(reference[:, token], moved_inside[:, token]), (
        f"raw sample {endpoint} is token {token}'s own newest sample and moved nothing -- the "
        f"perturbation never reached the module, so the bit-stability above proves nothing"
    )


def _perturb(x: torch.Tensor, index: int, amplitude: float = 1e3) -> torch.Tensor:
    """Return a copy of ``x`` with one raw sample displaced by ``amplitude``.

    A single-sample displacement rather than a resample of the whole tail, because the claim under
    test is about one boundary index. The amplitude is large so that a leak of any weight separates
    from float64 round-off by many orders of magnitude.

    Args:
        x: A raw batch, ``(B, L)``.
        index: The sample to displace.
        amplitude: How far to displace it.

    Returns:
        A new tensor shaped like ``x``.
    """
    perturbed = x.clone()
    perturbed[..., index] = perturbed[..., index] + amplitude
    return perturbed
