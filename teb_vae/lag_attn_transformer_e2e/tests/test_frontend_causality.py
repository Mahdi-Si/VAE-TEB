r"""The package's central test: token $t$ reads raw sample $16t + 15$ and not one sample more.

Everything else this package does is a rearrangement of code that already exists. This is the claim
it was built to make, and it is the claim that decides whether the source-conditioned KL readout
means what the model says it means -- a source-stream leak enters the posterior alone, so it
inflates $D_0 - D_1$, inflates $K$, and shifts the lag map.

Three properties of the assertion are deliberate.

**Bitwise, not thresholded.** A float32 threshold at this boundary is a statement about round-off
rather than about causality, which is the trap the sibling's source-window test documents hitting.
The probe runs in float64 and compares with ``torch.equal``, so a leak of any weight at all
separates from the noise floor by fifteen orders of magnitude rather than by a factor somebody had
to choose.

**Paired with movement.** A front end that returned zeros would be bit-stable at every cut. Every
assertion below therefore also requires the last token to have moved, which is what
:func:`assert_raw_causal` enforces on the caller's behalf.

**The negative control is a future-reading front end, built here and never in production code.** The
accident a real edit produces is a symmetrically padded FIR -- ``padding=(k-1)//2``, which is what
``nn.Conv1d`` does by default and what anybody reaching for the ``padding`` argument would write. A
centred *offset* would not be a valid control: it makes the token depend on raw samples
$\le 16t$, which is strictly more conservative, so a planted-broken model would pass the bitwise
half and the control would prove nothing.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from teb_vae.lag_attn_transformer_e2e.nets import frontend as frontend_module
from teb_vae.lag_attn_transformer_e2e.nets.frontend import CausalAntiAliasDecimate
from teb_vae.lag_attn_transformer_e2e.tests.conftest import (
    BATCH,
    MOVEMENT_TOL,
    SEQ_LEN,
    SHIPPED_KWARGS,
    TINY_KWARGS,
    assert_raw_causal,
    build_frontend,
    make_stub_batch,
    relative_change,
    resample_raw_after,
)

#: Tokens the probe is run at. More than one, because a stack could be causal at one offset and not
#: at another -- an alignment error that happened to cancel at $t = 0$ would be invisible.
PROBE_TOKENS = (6, 11)

#: Relative movement a change to a token's **single newest** raw sample must exceed. Four orders
#: below the shared ``MOVEMENT_TOL`` on purpose, and the reason is arithmetic rather than slack:
#: the cascade's leading tap is $16^{-4} \approx 1.5 \times 10^{-5}$, so that is the largest
#: movement one edge sample can produce. Nine orders above float64 round-off, which is what
#: separates "reads it faintly, as a low-pass must" from "does not read it".
ANCHOR_MOVEMENT_TOL = 1e-9


class _SymmetricallyPaddedDecimate(CausalAntiAliasDecimate):
    """The planted defect: the anti-alias filter padded on both sides instead of only on the left.

    Written here rather than offered as a constructor flag, because a switch in production code that
    exists only so a test can flip it is a second implementation nobody runs. This is a subclass of
    the real thing with one method replaced, so it inherits the same taps, the same stride and the
    same reach -- and therefore passes the reach guard, exactly as the real accident would.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter with the padding split evenly, then subsample at the same right offset."""
        left = (self.taps - 1) // 2
        padded = F.pad(x, (left, (self.taps - 1) - left))
        filtered = F.conv1d(padded, self.fir.to(dtype=x.dtype), groups=self.channels)
        return filtered[..., self.stride - 1 :: self.stride]


def _raw_pair(dtype: torch.dtype = torch.float64):
    """Return the stub batch's two raw signals and their shared weight, in ``dtype``.

    Both streams, because this package builds two independently parameterised front ends at
    identical settings and a probe run on one of them says nothing about the other's wiring.
    """
    batch = make_stub_batch(BATCH, SEQ_LEN)
    return batch.fhr.to(dtype), batch.up.to(dtype), batch.weight.to(dtype)


# ---------------------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("token", PROBE_TOKENS, ids=lambda t: f"t={t}")
@pytest.mark.parametrize("stream", ["target", "source"])
def test_a_token_reads_no_raw_sample_after_its_own_anchor(token, stream):
    r"""Resample every raw sample after $16t + 15$ and token $t$ must be bitwise identical.

    $16t + 15$ is ``TrimmedRawGeometry.n_raw(t)``: the newest raw sample the model's anchor $t$ is
    allowed to have seen. That the front end's decimation convention lands on exactly that index --
    rather than one either side of it -- is what makes the two conventions agree with no off-by-one
    to negotiate.
    """
    target_raw, source_raw, weight = _raw_pair()
    raw = target_raw if stream == "target" else source_raw
    net = build_frontend(TINY_KWARGS).double()

    with torch.no_grad():
        assert_raw_causal(
            lambda value: net(value, weight),
            raw,
            net.total_stride * (token + 1) - 1,
            net.total_stride,
            label=f"{stream} front end @ t={token}",
        )


@pytest.mark.parametrize("token", PROBE_TOKENS, ids=lambda t: f"t={t}")
def test_a_token_does_move_when_its_own_step_changes(token):
    r"""The other side of the boundary, in two magnitudes.

    Perturbing from $16t + 15$ onward -- one sample earlier than above -- must move token $t$ at
    all, or the front end would be discarding the newest quarter-second of every token and the test
    above would be certifying a stack that simply reads less than it may. Perturbing the token's
    whole step must move it *substantially*.

    The two bars differ by three orders of magnitude and that is a property of the design rather
    than a weakness of the test. Each stage's anti-alias filter puts the newest sample on its
    leading tap, $h_0 = 1/16$ at five binomial taps, so after four stages the token's newest raw
    sample carries roughly $16^{-4} \approx 1.5 \times 10^{-5}$ of it -- measured here at
    $1.3$ to $1.9 \times 10^{-5}$, against $3$ to $6 \times 10^{-3}$ for its sixteen samples
    together. A low-pass is supposed to weight the edge of its window lightly; what it may not do is
    weight anything past that edge at all, which is what the bitwise half asserts.
    """
    raw, _, weight = _raw_pair()
    net = build_frontend(TINY_KWARGS).double()
    anchor = net.total_stride * (token + 1) - 1

    with torch.no_grad():
        reference = net(raw, weight)
        from_anchor = net(resample_raw_after(raw, anchor - 1), weight)
        from_step = net(resample_raw_after(raw, net.total_stride * token - 1), weight)

    newest = relative_change(reference[:, token], from_anchor[:, token])
    own_step = relative_change(reference[:, token], from_step[:, token])
    assert newest > ANCHOR_MOVEMENT_TOL, (
        f"token {token} moved by only {newest:.3e} when its own newest raw sample {anchor} "
        f"changed, so the front end is not reading up to its anchor at all"
    )
    assert own_step > MOVEMENT_TOL, (
        f"token {token} moved by only {own_step:.3e} when its whole step was redrawn"
    )


def test_the_production_geometry_is_causal_too():
    """The smoke geometry shares the code but not the kernels, and it is the production kernels that
    reach $322$ raw samples. A stack can be causal at one width and not at another only through an
    arithmetic error, which is exactly the kind this probe exists to catch."""
    steps = int(SHIPPED_KWARGS["sequence_length"])
    stride = int(SHIPPED_KWARGS["raw_per_step"])
    net = build_frontend(SHIPPED_KWARGS).double()
    raw = torch.randn(1, steps * stride, dtype=torch.float64)
    weight = torch.ones(1, steps, dtype=torch.float64)

    with torch.no_grad():
        assert_raw_causal(
            lambda value: net(value, weight),
            raw,
            stride * 200 - 1,
            stride,
            label="production front end @ t=199",
        )


# ---------------------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------------------
def test_the_probe_rejects_a_symmetrically_padded_front_end(monkeypatch):
    """The control the whole file rests on. Split the anti-alias padding evenly -- the single most
    likely real edit, and one that changes no shape, no parameter count and no reach -- and the
    probe must reject it for reading its own future."""
    raw, _, weight = _raw_pair()
    monkeypatch.setattr(frontend_module, "CausalAntiAliasDecimate", _SymmetricallyPaddedDecimate)
    leaking = build_frontend(TINY_KWARGS).double()

    with torch.no_grad():
        with pytest.raises(AssertionError, match="reads its own future"):
            assert_raw_causal(
                lambda value: leaking(value, weight),
                raw,
                leaking.total_stride * 6 - 1,
                leaking.total_stride,
                label="symmetrically padded",
            )


def test_the_planted_defect_is_otherwise_indistinguishable(monkeypatch):
    """Why the control is the right one: it passes every *other* test the front end has. Shapes,
    reach and parameter count are all unchanged, so nothing but a causality probe could find it."""
    honest = build_frontend(TINY_KWARGS)
    monkeypatch.setattr(frontend_module, "CausalAntiAliasDecimate", _SymmetricallyPaddedDecimate)
    leaking = build_frontend(TINY_KWARGS)

    raw, _, weight = _raw_pair(torch.float32)
    with torch.no_grad():
        assert leaking(raw, weight).shape == honest(raw, weight).shape
    assert leaking.reach_samples == honest.reach_samples
    assert sum(p.numel() for p in leaking.parameters()) == sum(
        p.numel() for p in honest.parameters()
    )
