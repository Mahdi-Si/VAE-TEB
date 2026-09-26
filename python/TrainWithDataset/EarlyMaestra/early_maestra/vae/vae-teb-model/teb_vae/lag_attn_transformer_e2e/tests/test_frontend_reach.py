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

One composed bound lives here too, because only a raw input has it: the source front end's reach
composed with the source encoder's bounded window must stay inside the lag search range.
"""
from __future__ import annotations

import pytest
import torch

from teb_vae.lag_attn_transformer_e2e.nets.model import SeqVaeLagAttnTrfE2E
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    build_frontend,
)


def _budget(kwargs: dict) -> int:
    """The reach budget a geometry implies: ``warmup_period * raw_per_step`` raw samples."""
    return int(kwargs["warmup_period"]) * int(kwargs["raw_per_step"])


# ---------------------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------------------
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
    reach = build_frontend(TINY_KWARGS).reach_samples

    build_frontend(TINY_KWARGS, reach_budget=reach)
    with pytest.raises(ValueError, match="front end reaches"):
        build_frontend(TINY_KWARGS, reach_budget=reach - 1)


def test_the_composed_source_reach_stays_inside_the_lag_search_range():
    r"""A source state reaching further back than the lag range would already be doing the
    alignment the lag attention exists to do, and the reported lag would stop being a statement
    about where the coupling came from. At the production geometry the composed raw reach is
    $R_{\mathrm{frontend}} + r(R_U - 1)$ -- both reaches are counts, so the anchor token's own $r$
    samples overlap once -- and it must stay below the $r \cdot \mathrm{max\_lag}$ raw samples
    the lag search spans."""
    model = SeqVaeLagAttnTrfE2E(**SHIPPED_KWARGS)
    source_reach = model.source_encoder.receptive_field
    assert source_reach is not None, "the source encoder lost its bounded window"

    composed = model.source_frontend.reach_samples + model.raw_per_step * (source_reach - 1)

    assert composed < model.max_lag * model.raw_per_step


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
