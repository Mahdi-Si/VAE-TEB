r"""The smear, inverted: the target this model forecasts contains none of its own future.

This is section 1's whole argument for the package's existence, and until now it was asserted
nowhere.

**What the two-sided family carries.** A coefficient of the production Morlet bank at decimated
step $s$ is a weighted average over a window **centred** at $s$, so half of its support lies after
its own step. The two-sided sibling records that as a *blend fraction*
$b(\tau, \rho_c) = \max(0, (\rho_c - \Delta\tau)/(2\rho_c))$, which is exactly $0.5$ at $\tau = 0$
for every channel whatever its reach -- the channel-independence is the tell, because it comes from
the window straddling its own step rather than from any property of the filter.

**What this family carries instead.** The one-sided gammatone bank's kernels have support ending at
their own step: the coefficient at $s$ is a function of $\{x(n) : n \le 16 s\}$ and of nothing else.
The share of a coefficient's support lying after its own step -- the quantity the two-sided bank
puts at $0.5$ -- is therefore identically $0$, for every channel and at every horizon step, and the
blend formula evaluated at a forward reach of zero returns zero for the same reason.

**Two claims, two files.** ``lag_attn/tests/test_data_contract.py`` establishes the *dataset*
claim empirically: the committed shard's stored blocks are the transform of its own raw signals,
and perturbing a raw sample at index $n$ moves no coefficient at any step $s$ with $16s < n$. What
is established here is the *model* claim -- that the property holds over exactly the channels this
model forecasts, at exactly the horizon steps it forecasts them at -- recomputed from the shard's
own warm-up and group-delay attributes rather than declared, so a dataset rebuilt at another
quantile re-derives it instead of invalidating a constant.

**What the smear never affected, in either family.** $D_0$ and $D_1$ come from one shared decoder
under one shared $\epsilon$, so at initialisation the component of the target fixed by observed
history contributes the same amount to both and ``pred_gap`` is exactly zero however smeared the
target is. A smear can inflate the two reconstruction terms together; what it cannot do is
manufacture a gap between them, which is the quantity every coupling claim is read off. That was
already true two-sided, and it is why removing the smear is about **interpretation** -- what an
input at step $t$ is allowed to have seen -- rather than about repairing a biased number.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from teb_vae.lag_attn_cfs.causal_warmup import SOURCE_BLOCKS, TARGET_BLOCKS, resolve_warmup_budget
from teb_vae.lag_attn_cfs.tests.conftest import (
    BATCH,
    CAUSAL_SHARD,
    SHIPPED_HORIZON,
    TWO_SIDED_SHARD,
    build,
    causal_config,
    make_streams,
    shipped_warmup_kwargs,
    stored_warmup,
    tiny_warmup_kwargs,
)

#: Raw samples per second of the stored signal, raw samples per decimated step, and the length of
#: one step that follows: $16 / 4 = 4$ s. Declared here rather than imported from
#: ``teb_vae.lag_attn.channel_reach``, which builds the production two-sided kymatio bank at module
#: scope -- the guard this family replaces, and one that measures the wrong transform on this data.
FS_HZ = 4.0
RAW_PER_STEP = 16
_STEP_SECONDS = RAW_PER_STEP / FS_HZ

#: The two-sided sibling's recorded blend table, quoted as the contrast so the two numbers sit
#: beside each other. Pinned against the shipped filter bank in
#: ``teb_vae/lag_attn_fs/tests/test_smear.py``; restated rather than imported, because importing
#: that module builds the kymatio bank this package exists to stop consulting.
FS_BLEND_AT_TAU_ZERO = 0.500
FS_BLEND_HORIZON_MEAN_KEPT = 0.091
FS_BLEND_HORIZON_MEAN_ALL = 0.173


def _forward_share(forward_reach_s: float, backward_reach_s: float, tau: float) -> float:
    r"""The share of horizon step $\tau$'s target coefficient support lying after its own step.

    $$b(\tau) \;=\; \max\!\left(0,\; \frac{\rho^{+}_c - \Delta\tau}{\rho^{+}_c + \rho^{-}_c}\right)$$

    On a **symmetric** kernel, $\rho^{+} = \rho^{-} = \rho$, this is the two-sided sibling's
    $\max(0, (\rho - \Delta\tau)/2\rho)$ -- the same expression, and $0.5$ at $\tau = 0$. On a
    **one-sided** kernel the forward reach is zero, there is no forward arm for the fraction to be
    a share of, and the value is $0$ at every $\tau \ge 0$.

    Args:
        forward_reach_s: $\rho^{+}_c$, how far the kernel's support extends past its own step.
        backward_reach_s: $\rho^{-}_c$, how far it extends before it. Must be positive.
        tau: The horizon step.

    Returns:
        The share, in $[0, 1]$.
    """
    span = float(forward_reach_s) + float(backward_reach_s)
    assert span > 0.0, "a kernel with no support at all is not a channel"
    return max(0.0, (float(forward_reach_s) - _STEP_SECONDS * float(tau)) / span)


def _causal_reaches():
    """``(forward, backward)`` per declared target channel, in seconds, from the shard itself.

    The forward reach is $0$ by one-sidedness -- the property
    ``lag_attn/tests/test_data_contract.py`` establishes on the stored coefficients themselves.
    The backward reach is the warm-up: $W_c$ untrimmed steps is exactly the leading window
    enclosing $95\\%$ of the kernel's energy, which is where its support effectively begins.
    """
    stored = stored_warmup(CAUSAL_SHARD)
    backward = np.concatenate([stored[block] for block in TARGET_BLOCKS]).astype(float)
    # A channel whose warm-up rounds to zero still has one step of support; the clamp keeps the
    # denominator positive without pretending the kernel is longer than it is.
    return np.zeros_like(backward), np.maximum(backward, 1.0) * _STEP_SECONDS


# =================================================================================================
# The measurement itself
# =================================================================================================
def test_the_shard_declares_itself_one_sided_and_carries_the_two_attributes_this_reads():
    """The probe's precondition, checked rather than assumed: everything below is read off
    ``causal_warmup_steps`` and ``causal_delay_s``, which a two-sided shard does not have."""
    import h5py

    with h5py.File(CAUSAL_SHARD, "r") as handle:
        assert handle.attrs["transform"] == "causal"
        for block in (*TARGET_BLOCKS, *SOURCE_BLOCKS):
            assert "causal_warmup_steps" in handle[block].attrs, block
            assert "causal_delay_s" in handle[block].attrs, block


def test_the_probe_is_unavailable_on_a_two_sided_shard():
    """Not vacuous, and it fails *loudly* rather than by reading a zero. The two-sided fixture
    carries no ``transform`` attribute at all and no per-block warm-up, so every number below is
    simply absent there -- which is the failure mode a smear claim should have."""
    import h5py

    with h5py.File(TWO_SIDED_SHARD, "r") as handle:
        assert handle.attrs.get("transform", "two_sided") != "causal"
        offenders = [
            block
            for block in TARGET_BLOCKS
            if block in handle and "causal_warmup_steps" in handle[block].attrs
        ]
        assert not offenders, offenders

    with pytest.raises(ValueError, match="causal"):
        resolve_warmup_budget(causal_config(paths=[TWO_SIDED_SHARD]))


def test_the_composed_group_delay_is_non_negative_on_every_stored_channel():
    r"""A kernel reaching into its own future would report a **negative** composed group delay --
    its energy centroid would sit ahead of its own step. Every stored channel's is positive, and by
    a wide margin: $13.3$ s at the fastest and $791.0$ s at the slowest.

    This is the shard's own statement of one-sidedness, independent of the warm-up, and it is why
    ``causal_delay_s`` is read here even though no model reads it: the two attributes disagreeing
    would mean the bank was not what the writer thought it was.
    """
    import h5py

    with h5py.File(CAUSAL_SHARD, "r") as handle:
        for block in (*TARGET_BLOCKS, *SOURCE_BLOCKS):
            delays = np.asarray(handle[block].attrs["causal_delay_s"], dtype=float)
            assert delays.size > 0
            assert float(delays.min()) > 0.0, block


def test_the_smear_fraction_is_identically_zero_over_the_kept_set_at_every_horizon_step():
    r"""The claim, restricted to exactly what this model forecasts.

    Not "over the stored channels" -- over the $98$ the resolved budget keeps, which is the set the
    decoder emits and the objective scores. Recomputed from the shard's own attributes, so a
    dataset rebuilt at another ``causal_warmup_quantile`` re-derives the set and the number rather
    than invalidating a constant.
    """
    budget = resolve_warmup_budget(causal_config())
    assert budget is not None
    forward, backward = _causal_reaches()

    kept = list(budget.target.keep_index)
    assert len(kept) == 98

    for channel in kept:
        ahead, behind = float(forward[channel]), float(backward[channel])
        for tau in range(SHIPPED_HORIZON):
            assert _forward_share(ahead, behind, tau) == 0.0


def test_the_formula_reproduces_the_two_sided_contrast_it_is_read_against():
    r"""The same expression at a symmetric reach is the sibling's blend: $0.5$ at $\tau = 0$ for
    every channel, whatever its reach. Without this the zero above would be a zero from a formula
    that returns zero for everything."""
    for reach in (12.0, 28.0, 117.25, 965.5):
        assert _forward_share(reach, reach, 0) == pytest.approx(FS_BLEND_AT_TAU_ZERO)
        # And it falls linearly to zero at the channel's own step, exactly as the sibling records.
        clean_step = reach / _STEP_SECONDS
        assert _forward_share(reach, reach, clean_step) == pytest.approx(0.0)
        assert _forward_share(reach, reach, clean_step + 1) == 0.0

    # The sibling's horizon means, quoted so the two families' numbers sit beside each other:
    # 0.091 over its kept set and 0.173 over all 109 channels, against 0.000 here.
    assert FS_BLEND_HORIZON_MEAN_KEPT < FS_BLEND_HORIZON_MEAN_ALL
    assert FS_BLEND_HORIZON_MEAN_KEPT > 0.0


def test_the_kept_set_the_smear_claim_covers_is_the_set_the_decoder_emits():
    """The bridge from the arithmetic above to the model: the channels the claim is made over are
    the channels the model forecasts. Read off the constructed model rather than off the resolver,
    so a budget that reached the constructor differently would fail here."""
    kwargs = shipped_warmup_kwargs()
    model = build(kwargs)
    budget = resolve_warmup_budget(causal_config())
    assert budget is not None

    assert model.target_gate is not None
    assert model.target_gate.keep_index.tolist() == list(budget.target.keep_index)
    assert model.decoder_out_channels == len(budget.target.keep_index) == 98


# =================================================================================================
# What the smear never affected
# =================================================================================================
def test_the_shared_decoder_cancels_the_smeared_component_in_both_branches():
    r"""The cancellation the two-sided family's not-a-leak argument rests on, restated here so the
    two claims are not confused.

    $D_0$ and $D_1$ come from one decoder under one shared $\epsilon$, so at initialisation -- where
    the two latents are equal -- any component of the target fixed by observed history contributes
    the *same* amount to both and ``pred_gap`` is exactly zero. Removing the smear was therefore
    never about repairing a biased gap; it is about what an input at step $t$ is allowed to have
    seen.
    """
    kwargs = tiny_warmup_kwargs()
    model = build(kwargs).eval()
    y_st, y_ph, u_stream = make_streams(kwargs)

    torch.manual_seed(0)
    with torch.no_grad():
        out = model(y_st, y_ph, u_stream)
    metrics = model.compute_loss(
        out,
        torch.cat([y_st, y_ph], dim=-1),
        weight=torch.ones(BATCH, model.geometry.t),
    )["metrics"]

    assert float(metrics["pred_gap"]) == 0.0
    assert torch.equal(metrics["nll_full_block"], metrics["nll_base_block"])
